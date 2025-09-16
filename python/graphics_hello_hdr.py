import pathlib

from long_march import graphics
import numpy as np
import sys

def main():
    print("=== LongMarch Graphics Hello HDR Demo ===")
    print("Initializing graphics core...")

    core_settings = graphics.CoreSettings()
    print(f"Core Settings: {core_settings}")

    core = graphics.Core()
    print(f"Core Created: {core}")

    print("Initializing graphics device...")
    core.init_auto()
    print(f"Core Initialized: {core}")
    print(f"Device Name: {core.device_name()}")
    print(f"Ray Tracing Support: {core.ray_tracing_support()}")

    print("Creating HDR window...")
    window = core.create_window(1280, 720, "[Python] Graphics Hello HDR", resizable=True)
    window.set_hdr(True)  # Enable HDR
    print(f"Window Created: {window}")

    # Load shader file
    shader_path = "./shaders/hdr_shader.hlsl"
    print(f"Loading shader from: {shader_path}")
    print(f"Shader file exists: {pathlib.Path(shader_path).exists()}")

    with open(shader_path, "r") as f:
        shader_code = f.read()

    print("Compiling shaders...")
    vertex_shader = core.create_shader(shader_code, "VSMain", "vs_6_0")
    pixel_shader = core.create_shader(shader_code, "PSMain", "ps_6_0")
    print(f"Vertex Shader: {vertex_shader}")
    print(f"Pixel Shader: {pixel_shader}")

    print("Creating graphics program...")
    program = core.create_program([graphics.IMAGE_FORMAT_R32G32B32A32_SFLOAT], graphics.IMAGE_FORMAT_UNDEFINED)
    program.add_input_binding(24)  # sizeof(Vertex) = 6 floats * 4 bytes = 24 bytes
    program.add_input_attribute(0, graphics.INPUT_TYPE_FLOAT3, 0)  # position
    program.add_input_attribute(0, graphics.INPUT_TYPE_FLOAT3, 12)  # color
    program.bind_shader(vertex_shader, graphics.SHADER_TYPE_VERTEX)
    program.bind_shader(pixel_shader, graphics.SHADER_TYPE_PIXEL)
    program.finalize()
    print(f"Program Created: {program}")

    print("Preparing HDR geometry...")
    # Rectangle with HDR colors (values > 1.0)
    vertices = np.array([
        [-0.5, -0.1, 0.0, 0.0, 0.0, 0.0],  # Bottom left (black)
        [0.5, -0.1, 0.0, 3.0, 3.0, 3.0],   # Bottom right (bright white)
        [0.5, 0.1, 0.0, 3.0, 3.0, 3.0],    # Top right (bright white)
        [-0.5, 0.1, 0.0, 0.0, 0.0, 0.0],   # Top left (black)
    ], dtype=np.float32)

    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
    vertices_bytes = vertices.tobytes()
    indices_bytes = indices.tobytes()

    print(f"Vertex data: {len(vertices_bytes)} bytes ({len(vertices)} vertices)")
    print(f"Index data: {len(indices_bytes)} bytes ({len(indices)} indices)")

    print("Creating buffers...")
    vertex_buffer = core.create_buffer(len(vertices_bytes), graphics.BUFFER_TYPE_DYNAMIC)
    vertex_buffer.upload_data(vertices_bytes)
    index_buffer = core.create_buffer(len(indices_bytes), graphics.BUFFER_TYPE_DYNAMIC)
    index_buffer.upload_data(indices_bytes)
    print(f"Vertex Buffer: {vertex_buffer}")
    print(f"Index Buffer: {index_buffer}")

    print("Creating HDR frame buffer...")
    color_image = core.create_image(window.get_width(), window.get_height(), graphics.IMAGE_FORMAT_R32G32B32A32_SFLOAT)
    print(f"Color Image: {color_image}")

    def on_resize(width, height):
        print(f"Window resized to {width}x{height} - Recreating frame buffer")
        nonlocal color_image
        color_image = core.create_image(window.get_width(), window.get_height(), graphics.IMAGE_FORMAT_R32G32B32A32_SFLOAT)
        print(f"New Color Image: {color_image}")

    window.register_resize_event(on_resize)

    print("\n=== Starting HDR Render Loop ===")

    while not window.should_close():
        # Render frame
        cmd_ctx = core.create_command_context()
        cmd_ctx.cmd_clear_image(color_image, [0.0, 0.0, 0.0, 1.0])  # Black background
        cmd_ctx.cmd_begin_rendering([color_image], None)  # No depth buffer
        cmd_ctx.cmd_bind_program(program)
        cmd_ctx.cmd_bind_vertex_buffers(0, [vertex_buffer], [0])
        cmd_ctx.cmd_bind_index_buffer(index_buffer, 0)
        cmd_ctx.cmd_set_viewport(0, 0, window.get_width(), window.get_height(), 0.0, 1.0)
        cmd_ctx.cmd_set_scissor(0, 0, window.get_width(), window.get_height())
        cmd_ctx.cmd_set_primitive_topology(graphics.PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        cmd_ctx.cmd_draw_indexed(6, 1, 0, 0, 0)  # 2 triangles = 6 indices
        cmd_ctx.cmd_end_rendering()
        cmd_ctx.cmd_present(window, color_image)
        core.submit_command_context(cmd_ctx)
        window.poll_events()

    print(f"\n=== HDR Demo Complete ===")

if __name__ == '__main__':
    main()
