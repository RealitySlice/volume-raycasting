use bevy::{
    prelude::{Deref},
    app::{App, Plugin},
    
    asset::{Assets, AssetEvent, AssetResult, AssetServer,Handle, HandleUntyped,},
    core::cast_slice,
    core_pipeline::core_3d::{Camera3d, Camera3dBundle},
    DefaultPlugins,
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    ecs::{
        entity::{Entity}, query::{With, QueryState}, world::{FromWorld, World}, 
        system::{Commands, lifetimeless::{Read, SRes,SQuery}, Query, Res, ResMut, SystemParamItem}},
    math::{Mat4, Vec3},
    reflect::TypeUuid,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        camera::{Camera, ExtractedCamera, CameraDriverNode},
        Extract,
        renderer::{RenderDevice, RenderContext, RenderQueue},
        render_asset::RenderAssets,
        render_graph::{Node, NodeRunError, RenderGraph, RenderGraphContext,SlotInfo, SlotType},
        render_phase::{DrawFunctions, DrawFunctionId, EntityRenderCommand, PhaseItem, RenderCommand, TrackedRenderPass, RenderCommandResult, RenderPhase,},
        render_resource::{
            BlendState,BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry,
            BindingResource, BindGroupLayoutDescriptor, BindingType,
            Buffer, BufferUsages, BufferInitDescriptor, CachedRenderPipelineId, ColorWrites, 
            ColorTargetState, Extent3d, Face,
            FragmentState, IndexFormat, MultisampleState, PipelineCache,PipelineLayoutDescriptor,
            PrimitiveState, PrimitiveTopology, RenderPipeline, 
            RenderPipelineDescriptor, RenderPassDescriptor,
            Shader, ShaderModuleDescriptor, ShaderStages, SamplerBindingType, TextureDimension,
            TextureFormat, TextureSampleType, TextureUsages, TextureViewDimension,VertexAttribute,
            VertexBufferLayout, VertexFormat, VertexState, VertexStepMode,
        },
        texture::{BevyDefault, TextureCache, Image,},
        RenderApp,RenderStage,
        view::{ExtractedView, ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms},
    },
    transform::components::Transform,
    window::WindowDescriptor,
};

fn main() {
    App::new()
        .insert_resource(WindowDescriptor {
            title: format!(
                "{} {} - Volume Raycast",
                env!("CARGO_PKG_NAME"),
                env!("CARGO_PKG_VERSION")
            ),
            ..Default::default()
        })
        .add_plugins(DefaultPlugins)
        .add_plugin(FrameTimeDiagnosticsPlugin)
        .add_plugin(LogDiagnosticsPlugin::default())
        .add_plugin(RaycastPlugin)
        .add_startup_system(setup)
        .run();
}


fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
) {
    // In the compute_shader_game_of_life.rs example from the bevy repository, an image variable
    // is initialized using the new_fill() method of bevy::render::texture::Image.
    // Its texture descriptor.usage is then set(particularly, TextureUsages::STORAGE_BINDING).
    // Finally the image is stored as a resource into "images".
    let mut image = Image::new_fill(
        Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 256,
        },
        TextureDimension::D3,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8Unorm,
    );
    image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING;
    let image = images.add(image);

    commands.insert_resource(VolumeTexture(image));    
    // The Raycast Shader needs the camera's:
    // - global transform's translation vector
    // - projection matrix and its inverse
    
    // This data can be accessed through the bevy::render::entity::Camera3dBundle struct,
    // which provides:
    // - a bevy::render::camera::Camera that stores the Camera's projection matrix
    // and the method inverse() from the bevy::prelude::Mat4  struct
    // - the Camera's GlobalTransform struct. It stores a translation vector as bevy::prelude::Vec3
    
    commands.spawn_bundle(Camera3dBundle {
        transform: Transform::from_xyz(-2.0, 2.5, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..Default::default()
    });
}
struct RaycastPlugin;

impl Plugin for RaycastPlugin{
    fn build(&self, app: &mut App) {
        // Extract the camera resource from the main world into the render world
        // Extract the raycast texture resource from the main world into the render world
        // Store the Raycast Shader as a resource
        app.world.resource_mut::<Assets<Shader>>().set_untracked(
            RAYCAST_SHADER_HANDLE,
            Shader::from_wgsl(include_str!("../assets/shaders/raycast.wgsl")),
        );

        app.add_plugin(ExtractResourcePlugin::<VolumeTexture>::default());
        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<RaycastPipeline>()
            .add_system_to_stage(RenderStage::Extract, extract_raycast)
            .add_system_to_stage(RenderStage::Prepare, prepare_raycast)
            .add_system_to_stage(RenderStage::Queue, queue_raycast_volume_texture_bind_group)
            .add_system_to_stage(RenderStage::Queue, queue_raycast_view_bind_group)
            .add_system_to_stage(RenderStage::Queue, queue_raycast_bind_group);
            
        let mut render_graph = render_app.world.resource_mut::<RenderGraph>();
        let raycast_node = RaycastNode::new(&mut render_app.world);
        let camera_driver_node = CameraDriverNode::new(&mut render_app.world);
        render_graph.add_node("raycast_node", raycast_node);
        render_graph.add_node("camera_node", camera_driver_node);
        render_graph
            .add_node_edge(
                "camera_node",
                "raycast_node"
            )
            .unwrap();
    }


}
#[derive(Clone, Deref, ExtractResource)]
struct VolumeTexture(Handle<Image>);
struct VolumeTextureBindGroup(BindGroup);
pub struct RaycastPipeline{
    pub pipeline_id: CachedRenderPipelineId,
    pub texture_bind_group_layout: BindGroupLayout,
}

const RAYCAST_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 17343092250772987267);

// Define Raycast Pipeline
impl FromWorld for RaycastPipeline{
    // Create an instance of the RaycastPipeline trait using data from the supplied World.
    fn from_world(world: &mut World) -> Self{
        let render_device = world.get_resource::<RenderDevice>().unwrap();
        // Create Bind Group Layout for Volume Texture
        let texture_bind_group_layout = 
        render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("raycast texture bind group layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let mut pipeline_cache = world.resource_mut::<PipelineCache>();

        let pipeline_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            vertex: VertexState {
                shader: RAYCAST_SHADER_HANDLE.typed(),
                shader_defs: Vec::new(),
                entry_point: "vertex".into(),
                buffers: vec![VertexBufferLayout {
                    array_stride: 3 * 4,
                    step_mode: VertexStepMode::Vertex,
                    attributes: vec![
                        VertexAttribute {
                            format: VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0, // shader locations 0-2 are taken up by Position, Normal and UV attributes
                        },
                    ],
                }],
            },
            fragment: Some(FragmentState{
                shader: RAYCAST_SHADER_HANDLE.typed(),
                shader_defs: Vec::new(),
                entry_point: "fragment".into(),
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::bevy_default(),
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleStrip,
                cull_mode: Some(Face::Front),
                ..Default::default()
            },
            layout: Some(vec![texture_bind_group_layout.clone()]),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            label: Some("Raycast Pipeline".into()),

        });
        Self {
            pipeline_id,
            texture_bind_group_layout,
        }
    }
}

pub fn extract_raycast(
    mut commands: Commands,
    cameras_3d: Extract<Query<(Entity, &Camera), With<Camera3d>>>,
) {
    for (entity, camera) in cameras_3d.iter() {
        if camera.is_active {
            commands
                .get_or_spawn(entity)
                .insert(RenderPhase::<RaycastPhaseItem>::default());
        }
    }
}

fn prepare_raycast(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut gpu_mesh: ResMut<GpuMesh>
) {
    let vertices = [
        1., 1., 0., 0., 1., 0., 1., 1., 1., 0., 1., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1.,
        1., 0., 1., 0., 0., 1., 1., 1., 1., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0.,
    ];
    let vertex_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor{
        label: Some("Volume Vertex Buffer"),
        contents: cast_slice::<f32, _>(&vertices),
        usage: BufferUsages::VERTEX,
    });
    let vertex_count = vertices.len() / 3;
    gpu_mesh.buffer.write_buffer(&render_device, &render_queue);
}
fn prepare_view_uniforms(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut view_uniforms: ResMut<ViewUniforms>,
    views: Query<(Entity, &ExtractedView)>,
) {
    view_uniforms.uniforms.clear();
    for (entity, camera) in &views {
        let projection = camera.projection;
        let inverse_projection = projection.inverse();
        let view = camera.transform.compute_matrix();
        let inverse_view = view.inverse();
        let view_uniforms = ViewUniformOffset {
            offset: view_uniforms.uniforms.push(ViewUniform {
                view_proj: projection * inverse_view,
                inverse_view_proj: view * inverse_projection,
                view,
                inverse_view,
                projection,
                inverse_projection,
                world_position: camera.transform.translation(),
                width: camera.width as f32,
                height: camera.height as f32,
            }),
        };

        commands.entity(entity).insert(view_uniforms);
    }

    view_uniforms
        .uniforms
        .write_buffer(&render_device, &render_queue);
}

pub struct RaycastPhaseItem {
    pub draw_function: DrawFunctionId,
}

impl PhaseItem for RaycastPhaseItem {
    type SortKey = u32;

    #[inline]
    fn sort_key(&self) -> Self::SortKey {
        0
    }

    #[inline]
    fn draw_function(&self) -> DrawFunctionId {
        self.draw_function
    }
}
struct RaycastVolumeTextureBindGroup(BindGroup);

fn queue_raycast_bind_group(
    mut commands: Commands,
    raycast_pipeline: Res<RaycastPipeline>,
    raycast_volume_texture: Res<VolumeTexture>,
    render_device: Res<RenderDevice>,
) {
}

fn queue_raycast_view_bind_group(
    mut commands: Commands,
    raycast_pipeline: Res<RaycastPipeline>,
    raycast_volume_texture: Res<VolumeTexture>,
    render_device: Res<RenderDevice>,
) {
}

fn queue_raycast_volume_texture_bind_group(
    mut commands: Commands,
    raycast_pipeline: Res<RaycastPipeline>,
    raycast_volume_texture: Res<VolumeTexture>,
    render_device: Res<RenderDevice>,
) {
    let view = ;
    let bind_group = render_device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &raycast_pipeline.texture_bind_group_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::TextureView(&view.texture_view),
        }],
    });
    commands.insert_resource(VolumeTextureBindGroup(bind_group));
    //.write_buffer(&render_device, &render_queue);
}

pub struct RaycastNode {
    query: QueryState<
        (
            &'static RenderPhase<RaycastPhaseItem>,
            &'static ExtractedCamera,
            &'static ExtractedView,
            &'static Camera3d,
        ),
        With<ExtractedView>,
    >,
}
impl RaycastNode {
    pub const IN_VIEW: &'static str = "view";

    pub fn new(world: &mut World) -> Self {
        Self {
            query: world.query_filtered(),
        }
    }
}

impl Node for RaycastNode{
    fn input(&self) -> Vec<SlotInfo> {
        vec![SlotInfo::new(RaycastNode::IN_VIEW, SlotType::Entity)]
    }
    fn update(&mut self, _world: &mut World) {
        let pipeline = _world.resource::<RaycastPipeline>();
    }

    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let view_entity = graph.get_input_entity(Self::IN_VIEW)?;
        let (raycast_phase_item, camera, view, camera_3d) =
            match self.query.get_manual(world, view_entity) {
                Ok(query) => query,
                Err(_) => {
                    return Ok(());
                } // No window
            };

        let draw_functions = world.resource::<DrawFunctions<RaycastPhaseItem>>();

        let mut render_pass = render_context
        .command_encoder
        .begin_render_pass(&RenderPassDescriptor::default());
        let mut draw_functions = draw_functions.write();
        let mut tracked_pass = TrackedRenderPass::new(render_pass);
        if let Some(viewport) = camera.viewport.as_ref() {
            tracked_pass.set_camera_viewport(viewport);
        }
        // render_pass.set_bind_group(0,);
        // render_pass.set_bind_group(2, &volume_texture, &[]);
        // rpass.draw(0..self.vertex_count as _, 0..1);
        Ok(())
    }
}

// TODO: check compatibility with raycastpipeline definition
pub struct SetRaycastPipeline;

impl<P: PhaseItem> RenderCommand<P> for SetRaycastPipeline {
    type Param = (SRes<PipelineCache>, SRes<RaycastPipeline>);
    #[inline]
    fn render<'w>(
        _view: Entity,
        _item: &P,
        params: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let (pipeline_cache, raycast_pipeline) = params;
        if let Some(pipeline) = pipeline_cache
            .into_inner()
            .get_render_pipeline(raycast_pipeline.pipeline_id)
        {
            pass.set_render_pipeline(pipeline);
            RenderCommandResult::Success
        } else {
            RenderCommandResult::Failure
        }
    }
}
// TODO: check correctness
struct SetRaycastBindGroup<const I: usize>;
impl<const I: usize, P: PhaseItem> RenderCommand<P> for SetRaycastBindGroup<I> {
    type Param = SRes<GpuMesh>;

    #[inline]
    fn render<'w>(
        _view: Entity,
        _item: &P,
        gpu_mesh: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let gpu_mesh = gpu_mesh.into_inner();
        pass.set_bind_group(I, gpu_mesh.bind_group.as_ref().unwrap(), &[]);

        RenderCommandResult::Success
    }
}

// TODO: check correctness
#[derive(Debug, Clone)]
pub struct GpuMesh {
    /// Contains all attribute data for each vertex.
    vertex_buffer: Buffer,
    vertex_count: u32,
    primitive_topology: PrimitiveTopology,
    layout: VertexBufferLayout,
    bind_group: Option<BindGroup>,
}

// TODO: VERIFY CORRECTNESS
struct DrawRaycast;
impl EntityRenderCommand for DrawRaycast {
    type Param = SRes<GpuMesh>;
    #[inline]
    fn render<'w>(
        _view: Entity,
        item: Entity,
        meshes: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
            let gpu_mesh = meshes.into_inner();
            pass.set_vertex_buffer(0, gpu_mesh.vertex_buffer.slice(..));
            pass.draw(0..gpu_mesh.vertex_count, 0..1);

            RenderCommandResult::Success
    }
}
