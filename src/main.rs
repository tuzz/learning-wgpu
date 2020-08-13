use futures::executor::block_on;
use shaderc::{Compiler, ShaderKind::{Fragment, Vertex}};
use std::io::Cursor;

use wgpu::{
    Adapter, BackendBit, Color, CommandEncoder, CommandEncoderDescriptor,
    Device, DeviceDescriptor, LoadOp, PowerPreference, PresentMode, Queue,
    RenderPass, RenderPassColorAttachmentDescriptor, RenderPassDescriptor,
    RequestAdapterOptions, StoreOp, Surface, SwapChain, SwapChainDescriptor,
    SwapChainOutput, TextureFormat, TextureUsage, PipelineLayoutDescriptor,
    read_spirv, RenderPipelineDescriptor, ProgrammableStageDescriptor,
    RasterizationStateDescriptor, FrontFace, CullMode, ColorStateDescriptor,
    PrimitiveTopology, BlendDescriptor, ColorWrite, VertexStateDescriptor,
    IndexFormat, RenderPipeline
};

use winit::{
    dpi::PhysicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

fn main() {
    let event_loop = create_event_loop();
    let window = create_window(&event_loop);
    let size = window.inner_size();
    let surface = create_surface(&window);
    let adapter = request_adapter(&surface);
    let (device, queue) = request_device(&adapter);
    let mut swap_chain = create_swap_chain(&size, &surface, &device);
    let (vert, frag) = compile_shaders();
    let pipeline = create_render_pipeline(&device, &vert, &frag);

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::RedrawRequested(_) => {
                render_frame(&device, &queue, &mut swap_chain, &pipeline);
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

fn create_render_pipeline(device: &Device, vert: &[u32], frag: &[u32]) -> RenderPipeline {
    let layout_descriptor = PipelineLayoutDescriptor { bind_group_layouts: &[] };
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
        vertex_buffers: &[],
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

fn render_frame(device: &Device, queue: &Queue, mut swap_chain: &mut SwapChain, pipeline: &RenderPipeline) {
    let frame = next_frame(&mut swap_chain);
    let mut encoder = command_encoder(&device);
    let mut render_pass = begin_render_pass(&mut encoder, &frame);

    render_pass.set_pipeline(&pipeline);
    render_pass.draw(0..3, 0..1);

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