use bevy::{
    ecs::world::{FromWorld, World},
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        entity::entity::Camera3dBundle,
        camera::{Camera},
        render_graph::{Node, RenderGraph},
        resource::{
            BindGroup, BindGroupLayout, BindGroupLayoutEntry,BindGroupLayoutDescriptor,BindingType, 
            RenderPipeline, ShaderModuleDescriptor,ShaderStages,TextureSampleType,
            TextureViewDimension,
        },
        renderer::{RenderDevice},
    },
};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_startup_system(setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Images>>,
) {
    //In the compute_shader_game_of_life.rs example, an image variable
    //is initialized using the new_fill() method of bevy::render::texture::Image
    //its texture descriptor.usage is set(particularly, TextureUsages::STORAGE_BINDING)
    //finally the image is stored as a resource into "images"
    //
    //I
    // 
    //spawn Camera Bundle 
    // by using a bundle we have access
    // commands.spawn_bundle(Camera3dBundle {
    //     transform: Transform::from_xyz(-2.0, 2.5, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
    //     ..default()
    // });
}

pub struct RaycastPipeline{
    pipeline: RenderPipeline,

    pub texture_bind_group_layout: BindGroupLayout,
}

// Define Raycast Pipeline
impl FromWorld for RaycastPipeline{
    fn from_world(world: &mut World) -> Self{
        let render_device = world.get_resource::<RenderDevice>().unwrap();
        let shader_source = ShaderSource::Wgsl(include_str!("raycast.wgsl").into());
        let shader_module = render_device.create_shader_module(ShaderModuleDescriptor {
            label: Some("raycast shader"),
            source: shader_source,
        });
        // Create Bind Group Layout for Volume Texture
        let texture_bind_group_layout = 
        render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("raycast transform bind group layout"),
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
    }
}

pub struct RaycastNode{

}

impl Node for RaycastNode{

    fn update(&mut self, _world: &mut World) {
        let pipeline = world.resource::<RaycastPipeline>();
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

struct RaycastPlugin{

}

#[derive(Clone, Deref, ExtractResource)]
struct VolumeTexture(Handle<Image>);

struct VolumeTextureBindGroup(BindGroup);

fn queue_bind_group(
    mut commands: Commands,
    pipeline: Res<Raycastipeline>,
    volume_texture: Res<VolumeTexture>,
    render_device: Res<RenderDevice>,
) {
}

// STEPS(uncertain): 
// Initialize the 
// Query the transform of the camera component "View" transforms to Raycast Node.
impl Plugin for RaycastPlugin{
    fn build(&self, app: &mut App) {
        // Extract the camera resource from the main world into the render world
        // Extract the raycast texture resource from the main world into the render world
        app.add_plugin(ExtractResourcePlugin::<VoumeTexture>::default());
        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<RaycastPipeline>()
            .add_system_to_stage(RenderStage::Queue, queue_bind_group);

        let mut render_graph = render_app.world.resource_mut::<RenderGraph>();
        render_graph.add_node("raycast", RaycastNode::default());

        // render_graph
        //     .add_node_edge(
        //         "camera",
        //         bevy::render::main_graph::node::CAMERA_DRIVER,
        //     )
        //     .unwrap();
    }


}

