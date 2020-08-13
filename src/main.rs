use bytemuck::cast_slice;
use futures::executor::block_on;
use image::{GenericImageView, load_from_memory, DynamicImage};
use shaderc::{Compiler, ShaderKind::{Fragment, Vertex}};
use std::{io::Cursor, mem};

use wgpu::{
    Adapter, BackendBit, Color, CommandEncoder, CommandEncoderDescriptor,
    Device, DeviceDescriptor, LoadOp, PowerPreference, PresentMode, Queue,
    RenderPass, RenderPassColorAttachmentDescriptor, RenderPassDescriptor,
    RequestAdapterOptions, StoreOp, Surface, SwapChain, SwapChainDescriptor,
    SwapChainOutput, TextureFormat, TextureUsage, PipelineLayoutDescriptor,
    read_spirv, RenderPipelineDescriptor, ProgrammableStageDescriptor,
    RasterizationStateDescriptor, FrontFace, CullMode, ColorStateDescriptor,
    PrimitiveTopology, BlendDescriptor, ColorWrite, VertexStateDescriptor,
    IndexFormat, RenderPipeline, BufferUsage, VertexBufferDescriptor, Buffer,
    InputStepMode, VertexFormat, BufferAddress, VertexAttributeDescriptor,
    Extent3d, TextureDimension, TextureDescriptor, Texture, BufferCopyView,
    TextureCopyView, Origin3d, AddressMode, CompareFunction, FilterMode,
    SamplerDescriptor, Sampler, BindGroupLayoutDescriptor, BindingType,
    ShaderStage, BindGroupLayoutEntry, TextureComponentType, BindGroupLayout,
    TextureViewDimension, Binding, BindingResource, TextureView,
    BindGroupDescriptor, BindGroup,
};

use winit::{
    dpi::PhysicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Point {
    position: [f32; 3],
    tex_coords: [f32; 2],
}

unsafe impl bytemuck::Pod for Point {}
unsafe impl bytemuck::Zeroable for Point {}

const POINTS: &[Point] = &[
    Point { position: [0.0, 0.5, 0.0], tex_coords: [1.0, 0.0] },
    Point { position: [-0.5, -0.5, 0.0], tex_coords: [0.0, 1.0] },
    Point { position: [0.5, -0.5, 0.0], tex_coords: [0.0, 0.0] },
];

fn main() {
    let event_loop = create_event_loop();
    let window = create_window(&event_loop);
    let size = window.inner_size();
    let surface = create_surface(&window);
    let adapter = request_adapter(&surface);
    let (device, queue) = request_device(&adapter);
    let mut swap_chain = create_swap_chain(&size, &surface, &device);
    let (buffer, descriptor) = create_buffer(&device);
    let image = load_image(include_bytes!("happy-tree.png"));
    let texture = create_texture(&device, &image);
    let (sampler, view) = create_texture_sampler(&device, &texture);
    let (bind_group, layout) = create_bind_group(&device, &sampler, &view);
    let (vert, frag) = compile_shaders();
    let pipeline = create_render_pipeline(&device, &layout, &vert, &frag, descriptor);

    copy_image_to_texture(&device, &queue, &image, &texture);

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::RedrawRequested(_) => {
                render_frame(&device, &queue, &mut swap_chain, &buffer, &bind_group, &pipeline);
            },
            Event::MainEventsCleared => {
                window.request_redraw();
            },
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::Resized(size) => {
                    swap_chain = create_swap_chain(&size, &surface, &device);
                },
                WindowEvent::ScaleFactorChanged { new_inner_size: size, .. } => {
                    swap_chain = create_swap_chain(&size, &surface, &device);
                },
                WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                },
                _ => {},
            },
            _ => {},
        }
    });
}

fn create_event_loop() -> EventLoop<()> {
    EventLoop::new()
}

fn create_window(event_loop: &EventLoop<()>) -> Window {
    WindowBuilder::new().build(event_loop).unwrap()
}

fn create_surface(window: &Window) -> Surface {
    Surface::create(window)
}

fn request_adapter(surface: &Surface) -> Adapter {
    let options = RequestAdapterOptions {
        power_preference: PowerPreference::Default,
        compatible_surface: Some(surface)
    };

    let future = Adapter::request(&options, BackendBit::PRIMARY);

    block_on(future).unwrap()
}

fn request_device(adapter: &Adapter) -> (Device, Queue) {
    let descriptor = DeviceDescriptor::default();
    let future = adapter.request_device(&descriptor);

    block_on(future)
}

fn create_swap_chain(window_size: &PhysicalSize<u32>, surface: &Surface, device: &Device) -> SwapChain {
    let descriptor = SwapChainDescriptor {
        width: window_size.width,
        height: window_size.height,
        usage: TextureUsage::OUTPUT_ATTACHMENT, // Writes to the screen
        format: TextureFormat::Bgra8UnormSrgb,  // Guaranteed to be supported
        present_mode: PresentMode::Fifo,        // Enable vsync
    };

    device.create_swap_chain(surface, &descriptor)
}

fn create_buffer(device: &Device) -> (Buffer, VertexBufferDescriptor) {
    let buffer = device.create_buffer_with_data(cast_slice(POINTS), BufferUsage::VERTEX);

    let descriptor = VertexBufferDescriptor {
        stride: mem::size_of::<Point>() as wgpu::BufferAddress,
        step_mode: InputStepMode::Vertex,
        attributes: &[
            VertexAttributeDescriptor {
                offset: 0,
                shader_location: 0,
                format: VertexFormat::Float3,
            },
            VertexAttributeDescriptor {
                offset: mem::size_of::<[f32; 3]>() as BufferAddress,
                shader_location: 1,
                format: VertexFormat::Float2,
            },
        ],
    };

    (buffer, descriptor)
}

fn load_image(bytes: &[u8]) -> DynamicImage {
    load_from_memory(bytes).unwrap()
}

fn create_texture(device: &Device, image: &DynamicImage) -> Texture {
    let (width, height) = image.dimensions();

    let descriptor = TextureDescriptor {
        size: Extent3d { width, height, depth: 1 },
        array_layer_count: 1,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8UnormSrgb,
        usage: TextureUsage::SAMPLED | TextureUsage::COPY_DST,
        label: None,
    };

    device.create_texture(&descriptor)
}

fn copy_image_to_texture(device: &Device, queue: &Queue, image: &DynamicImage, texture: &Texture) {
    let (width, height) = image.dimensions();
    let data = image.as_rgba8().unwrap();

    let mut encoder = command_encoder(&device);
    let buffer = device.create_buffer_with_data(&data, wgpu::BufferUsage::COPY_SRC);

    let buffer_copy = BufferCopyView {
        buffer: &buffer,
        offset: 0,
        bytes_per_row: 4 * width,
        rows_per_image: height,
    };

    let texture_copy = TextureCopyView {
        texture: &texture,
        mip_level: 0,
        array_layer: 0,
        origin: Origin3d::ZERO,
    };

    let extent = Extent3d { width, height, depth: 1 };

    encoder.copy_buffer_to_texture(buffer_copy, texture_copy, extent);
    queue.submit(&[encoder.finish()]);
}

fn create_texture_sampler(device: &Device, texture: &Texture) -> (Sampler, TextureView) {
    let texture_view = texture.create_default_view();

    let descriptor = SamplerDescriptor {
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Nearest,
        mipmap_filter: FilterMode::Nearest,
        lod_min_clamp: -100.0,
        lod_max_clamp: 100.0,
        compare: CompareFunction::Always,
    };

    (device.create_sampler(&descriptor), texture_view)
}

fn create_bind_group(device: &Device, sampler: &Sampler, view: &TextureView) -> (BindGroup, BindGroupLayout) {
    let layout_descriptor = BindGroupLayoutDescriptor {
        label: None,
        bindings: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStage::FRAGMENT,
                ty: BindingType::SampledTexture {
                    multisampled: false,
                    dimension: TextureViewDimension::D2,
                    component_type: TextureComponentType::Uint,
                },
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStage::FRAGMENT,
                ty: wgpu::BindingType::Sampler {
                    comparison: false,
                },
            },
        ],
    };

    let layout = device.create_bind_group_layout(&layout_descriptor);

    let descriptor = BindGroupDescriptor {
        label: None,
        layout: &layout,
        bindings: &[
            Binding { binding: 0, resource: BindingResource::TextureView(view) },
            Binding { binding: 1, resource: BindingResource::Sampler(sampler) },
        ],
    };

    (device.create_bind_group(&descriptor), layout)
}

fn compile_shaders() -> (Vec<u32>, Vec<u32>) {
    let mut compiler = Compiler::new().unwrap();

    let source = include_str!("shader.vert");
    let spirv = compiler.compile_into_spirv(source, Vertex, "", "main", None).unwrap();
    let vert = read_spirv(Cursor::new(spirv.as_binary_u8())).unwrap();

    let source = include_str!("shader.frag");
    let spirv = compiler.compile_into_spirv(source, Fragment, "", "main", None).unwrap();
    let frag = read_spirv(Cursor::new(spirv.as_binary_u8())).unwrap();

    (vert, frag)
}

fn create_render_pipeline(device: &Device, layout: &BindGroupLayout, vert: &[u32], frag: &[u32], buffer_descriptor: VertexBufferDescriptor) -> RenderPipeline {
    let layout_descriptor = PipelineLayoutDescriptor { bind_group_layouts: &[layout] };
    let layout = &device.create_pipeline_layout(&layout_descriptor);

    let module = &device.create_shader_module(&vert);
    let vertex_stage = ProgrammableStageDescriptor { module, entry_point: "main" };

    let module = &device.create_shader_module(&frag);
    let fragment_stage = Some(ProgrammableStageDescriptor { module, entry_point: "main" });

    let rasterization_state = Some(RasterizationStateDescriptor {
        front_face: FrontFace::Ccw,
        cull_mode: CullMode::Back,
        depth_bias: 0,
        depth_bias_slope_scale: 0.0,
        depth_bias_clamp: 0.0,
    });

    let color_descriptor = ColorStateDescriptor {
        format: TextureFormat::Bgra8UnormSrgb,
        color_blend: BlendDescriptor::REPLACE,
        alpha_blend: BlendDescriptor::REPLACE,
        write_mask: ColorWrite::ALL,
    };

    let vertex_state = VertexStateDescriptor {
        index_format: IndexFormat::Uint16,
        vertex_buffers: &[buffer_descriptor],
    };

    device.create_render_pipeline(&RenderPipelineDescriptor {
        layout,
        vertex_stage,
        fragment_stage,
        rasterization_state,
        color_states: &[color_descriptor],
        vertex_state,
        primitive_topology: PrimitiveTopology::TriangleList,
        alpha_to_coverage_enabled: false,
        depth_stencil_state: None,
        sample_count: 1,
        sample_mask: !0,
    })
}

fn render_frame(device: &Device, queue: &Queue, mut swap_chain: &mut SwapChain, buffer: &Buffer, bind_group: &BindGroup, pipeline: &RenderPipeline) {
    let frame = next_frame(&mut swap_chain);
    let mut encoder = command_encoder(&device);
    let mut render_pass = begin_render_pass(&mut encoder, &frame);

    render_pass.set_pipeline(pipeline);
    render_pass.set_bind_group(0, bind_group, &[]);
    render_pass.set_vertex_buffer(0, &buffer, 0, 0);
    render_pass.draw(0..POINTS.len() as u32, 0..1);

    drop(render_pass);
    queue.submit(&[encoder.finish()]);
}

fn next_frame(swap_chain: &mut SwapChain) -> SwapChainOutput {
    swap_chain.get_next_texture().unwrap()
}

fn command_encoder(device: &Device) -> CommandEncoder {
    let descriptor = CommandEncoderDescriptor { label: None };

    device.create_command_encoder(&descriptor)
}

fn begin_render_pass<'a>(encoder: &'a mut CommandEncoder, frame: &'a SwapChainOutput) -> RenderPass<'a> {
    let color_descriptor = RenderPassColorAttachmentDescriptor {
        attachment: &frame.view,
        resolve_target: None,
        load_op: LoadOp::Clear,
        store_op: StoreOp::Store,
        clear_color: Color { r: 0.1, g: 0.2, b: 0.3, a: 1.0 },
    };

    let render_descriptor = RenderPassDescriptor {
        depth_stencil_attachment: None,
        color_attachments: &[color_descriptor],
    };

    encoder.begin_render_pass(&render_descriptor)
}
