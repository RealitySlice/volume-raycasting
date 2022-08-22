use bevy::{
    prelude::{Deref},
    app::{App, Plugin},
    asset::{ Assets, AssetEvent, AssetResult, AssetServer,Handle},
    core_pipeline::core_3d::Camera3dBundle,
    DefaultPlugins,
    ecs::{world::{FromWorld, World}, system::{Commands, Res, ResMut}},
    math::{Mat4, Vec3},
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        camera::{Camera},
        renderer::{RenderDevice,RenderContext},
        render_asset::RenderAssets,
        render_graph::{Node, NodeRunError, RenderGraph, RenderGraphContext},
        render_resource::{
            BlendState,BindGroup, BindGroupLayout, BindGroupLayoutEntry,BindGroupLayoutDescriptor, BindingType,
            CachedRenderPipelineId, ColorWrites, ColorTargetState, Face,
            FragmentState, MultisampleState, PipelineCache,PipelineLayoutDescriptor,
            PrimitiveState, PrimitiveTopology, RenderPipeline, RenderPipelineDescriptor,
            Shader, ShaderModuleDescriptor, ShaderStages, SamplerBindingType,
            TextureFormat, TextureSampleType, TextureViewDimension,VertexAttribute,
            VertexBufferLayout, VertexFormat, VertexState, VertexStepMode,
        },
        texture::{BevyDefault, TextureCache, Image},
        RenderApp,RenderStage
    },
    transform::components::Transform,
};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
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
    
    // The Raycast Shader needs the camera's:
    // - global transform's translation vector
    // - projection matrix and its inverse
    
    // This data can be accessed through the bevy::render::entity::Camera3dBundle struct,
    // which provides:
    // - a bevy::render::camera::Camera that stores the Camera's projection matrix
    // and the method inverse() from the bevy::prelude::Mat4  struct
    // - the Camera's GlobalTransform struct. It stores a translation vector as bevy::prelude::Vec3
    
    // By spawning the Camera3dBundle we ensure the co-presence of the components of the Camera entity we seek to query.
    // To send the desired data to the GPU, we must perform the query during the Extract stage of our
    // bevy::render::RenderStage 
    
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
        app.add_plugin(ExtractResourcePlugin::<VolumeTexture>::default());
        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<RaycastPipeline>()
            .add_system_to_stage(RenderStage::Queue, queue_bind_group);

        let mut render_graph = render_app.world.resource_mut::<RenderGraph>();
        //render_graph.add_node("raycast", RaycastNode::default());
        // How and where do I add the camera node to my render graph 
        // for it to be a dependency of the raycast node?
        // How do I set the camera view transforms to be in the input slots of the raycast node?
        // render_graph
        //     .add_node_edge(
        //         "camera",
        //         bevy::render::main_graph::node::CAMERA_DRIVER,
        //     )
        //     .unwrap();
    }


}
#[derive(Clone, Deref, ExtractResource)]
struct VolumeTexture(Handle<Image>);
struct VolumeTextureBindGroup(BindGroup);
pub struct RaycastPipeline{
    pub pipeline_id: CachedRenderPipelineId,
    pub texture_bind_group_layout: BindGroupLayout,
}
// Define Raycast Pipeline

impl FromWorld for RaycastPipeline{
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
        let shader = world
        .resource::<AssetServer>()
        .load("/shaders/raycast.wgsl");
        let mut pipeline_cache = world.resource_mut::<PipelineCache>();

        //TODO: figure out how to use the pipeline
        let pipeline_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            vertex: VertexState {
                shader: shader.clone(),
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
                shader: shader.clone(),
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
struct GameOfLifeImageBindGroup(BindGroup);

fn queue_bind_group(
    mut commands: Commands,
    pipeline: Res<RaycastPipeline>,
    gpu_images: Res<RenderAssets<Image>>,
    game_of_life_image: Res<VolumeTexture>,
    render_device: Res<RenderDevice>,
) {
    
}
pub struct RaycastNode{
}

impl Node for RaycastNode{

    fn update(&mut self, _world: &mut World) {
        let pipeline = _world.resource::<RaycastPipeline>();
    }

    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        Ok(())
    }
}



