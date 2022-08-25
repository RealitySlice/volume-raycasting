use bevy::{
    prelude::{Deref},
    app::{App, Plugin},
    
    asset::{Assets, AssetEvent, AssetResult, AssetServer,Handle, HandleUntyped,},
    core_pipeline::core_3d::{Camera3dCamera3dBundle},
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

        renderer::{RenderDevice, RenderContext, RenderQueue},
        render_asset::RenderAssets,
        render_graph::{Node, NodeRunError, RenderGraph, RenderGraphContext,SlotInfo, SlotType},
        render_phase::{EntityRenderCommand, PhaseItem, TrackedRenderPass, RenderCommand, RenderCommandResult},
        render_resource::{
            BlendState,BindGroup, BindGroupLayout, BindGroupLayoutEntry,BindGroupLayoutDescriptor, BindingType,
            Buffer, BufferInitDescriptor, CachedRenderPipelineId, ColorWrites, 
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
        view::{ExtractedView, ViewTarget},
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

        //TODO: figure out how to use the pipeline
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
fn extract_raycast(mut commands: Commands, mut cubes: Query<(Entity)>) {

}
fn prepare_raycast(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
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
}

fn prepare_volume(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {

}
pub struct RaycastNode {
    query: QueryState<
        (
            &'static ExtractedCamera,
            &'static ExtractedView,
        ),
        With<ExtractedView>,
    >,
}
impl RaycastNode {
    pub const RAYCAST_NODE: &'static str = "raycast_node";

    pub fn new(world: &mut World) -> Self {
        Self {
            query: world.query_filtered(),
        }
    }
}
impl Node for RaycastNode{
    fn input(&self) -> Vec<SlotInfo> {
        vec![SlotInfo::new(CameraDriverNode::IN_VIEW, SlotType::Entity)]
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
        let (camera,target) =
            match self.query.get_manual(world, view_entity) {
                Ok(query) => query,
                Err(_) => {
                    return Ok(());
                } // No window
            };
        let mut render_pass = render_context
        .command_encoder
        .begin_render_pass(&RenderPassDescriptor::default());
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

struct SetRaycastBindGroup<const I: usize>;
impl<const I: usize, P: PhaseItem> RenderCommand<P> for SetRaycastBindGroup<I> {
    type Param = SRes<GpuMesh>;

    #[inline]
    fn render<'w>(
        _view: Entity,
        _item: &P,
        gpu_quads: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let gpu_quads = gpu_quads.into_inner();
        pass.set_bind_group(I, gpu_quads.bind_group.as_ref().unwrap(), &[]);

        RenderCommandResult::Success
    }
}

struct SetRaycastViewBindGroup<const I: usize>;
impl<const I: usize, P: PhaseItem> RenderCommand<P> for SetRaycastViewBindGroup<I> {
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

/// The GPU-representation of a [`Mesh`].
/// Consists of a vertex data buffer and an optional index data buffer.
#[derive(Debug, Clone)]
pub struct GpuMesh {
    /// Contains all attribute data for each vertex.
    pub vertex_buffer: Buffer,
    pub buffer_info: GpuBufferInfo,
    pub primitive_topology: PrimitiveTopology,
    pub layout: MeshVertexBufferLayout,
}

pub struct Mesh {
    primitive_topology: PrimitiveTopology,
}

/// The index/vertex buffer info of a [`GpuMesh`].
#[derive(Debug, Clone)]
pub enum GpuBufferInfo {
    Indexed {
        /// Contains all index data of a mesh.
        buffer: Buffer,
        count: u32,
        index_format: IndexFormat,
    },
    NonIndexed {
        vertex_count: u32,
    },
}
struct DrawRaycast;
impl EntityRenderCommand for DrawRaycast {
    type Param = SRes<GpuMesh>;
    #[inline]
    fn render<'w>(
        _view: Entity,
        item: Entity,
        (meshes, mesh_query): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let mesh_handle = mesh_query.get(item).unwrap();
        if let Some(gpu_mesh) = meshes.into_inner().get(mesh_handle) {
            pass.set_vertex_buffer(0, gpu_mesh.vertex_buffer.slice(..));
            match &gpu_mesh.buffer_info {
                GpuBufferInfo::Indexed {
                    buffer,
                    index_format,
                    count,
                } => {
                    pass.set_index_buffer(buffer.slice(..), 0, *index_format);
                    pass.draw_indexed(0..count, 0, 0..1);
                }
                GpuBufferInfo::NonIndexed { vertex_count } => {
                    pass.draw(0..vertex_count, 0..1);
                }
            }
            RenderCommandResult::Success
        } else {
            RenderCommandResult::Failure
        }
    }
}
