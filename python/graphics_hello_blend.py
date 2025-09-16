import pathlib

from long_march import graphics
import numpy as np
import sys

def main():
    print("=== LongMarch Graphics Hello Blend Demo ===")
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

    print("Creating window...")
    window = core.create_window(1280, 720, "[Python] Graphics Hello Blending", resizable=True)
    print(f"Window Created: {window}")

    # Load shader file
    shader_path = "./shaders/blend_shader.hlsl"
    print(f"Loading shader from: {shader_path}")
    print(f"Shader file exists: {pathlib.Path(shader_path).exists()}")

    with open(shader_path, "r") as f:
        shader_code = f.read()

    print("Compiling shaders...")
    vertex_shader = core.create_shader(shader_code, "VSMain", "vs_6_0")
    pixel_shader = core.create_shader(shader_code, "PSMain", "ps_6_0")
    print(f"Vertex Shader: {vertex_shader}")
    print(f"Pixel Shader: {pixel_shader}")

    print("Creating graphics program with blending...")
    program = core.create_program([graphics.IMAGE_FORMAT_R32G32B32A32_SFLOAT], graphics.IMAGE_FORMAT_UNDEFINED)
    program.add_input_binding(28)  # sizeof(Vertex) = 7 floats * 4 bytes = 28 bytes
    program.add_input_attribute(0, graphics.INPUT_TYPE_FLOAT3, 0)  # position
    program.add_input_attribute(0, graphics.INPUT_TYPE_FLOAT4, 12)  # color

    # Create blend state for alpha blending
    blend_state = graphics.BlendState(True)  # Enable blending
    program.set_blend_state(0, blend_state)  # Set blend state for render target 0

    program.bind_shader(vertex_shader, graphics.SHADER_TYPE_VERTEX)
    program.bind_shader(pixel_shader, graphics.SHADER_TYPE_PIXEL)
    program.finalize()
    print(f"Program Created: {program}")

    print("Preparing geometry with alpha blending...")
    # Two triangles with alpha values for blending
    vertices = np.array([
        # First triangle (semi-transparent)
        [0.0, 0.5, 0.0, 0.0, 1.0, 1.0, 0.5],    # Top vertex (cyan, 50% alpha)
        [-0.5, -0.5, 0.0, 1.0, 1.0, 0.0, 0.5],  # Bottom left (yellow, 50% alpha)
        [0.5, -0.5, 0.0, 1.0, 0.0, 1.0, 0.5],   # Bottom right (magenta, 50% alpha)

        # Second triangle (opaque, smaller)
        [0.0, 0.25, 0.0, 1.0, 0.0, 0.0, 1.0],   # Top vertex (red, 100% alpha)
        [-0.3, -0.35, 0.0, 0.0, 0.0, 1.0, 1.0], # Bottom left (blue, 100% alpha)
        [0.3, -0.35, 0.0, 0.0, 1.0, 0.0, 1.0],  # Bottom right (green, 100% alpha)
    ], dtype=np.float32)

    indices = np.array([0, 1, 2], dtype=np.uint32)
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

    print("Creating frame buffer...")
    color_image = core.create_image(window.get_width(), window.get_height(), graphics.IMAGE_FORMAT_R32G32B32A32_SFLOAT)
    print(f"Color Image: {color_image}")

    def on_resize(width, height):
        print(f"Window resized to {width}x{height} - Recreating frame buffer")
        nonlocal color_image
        color_image = core.create_image(window.get_width(), window.get_height(), graphics.IMAGE_FORMAT_R32G32B32A32_SFLOAT)
        print(f"New Color Image: {color_image}")

    window.register_resize_event(on_resize)

    print("\n=== Starting Render Loop ===")

    while not window.should_close():
        # Render frame
        cmd_ctx = core.create_command_context()
        cmd_ctx.cmd_clear_image(color_image, [0.6, 0.7, 0.8, 1.0])  # Light blue background
        cmd_ctx.cmd_begin_rendering([color_image], None)  # No depth buffer
        cmd_ctx.cmd_bind_program(program)
        cmd_ctx.cmd_bind_vertex_buffers(0, [vertex_buffer], [0])
        cmd_ctx.cmd_bind_index_buffer(index_buffer, 0)
        cmd_ctx.cmd_set_viewport(0, 0, window.get_width(), window.get_height(), 0.0, 1.0)
        cmd_ctx.cmd_set_scissor(0, 0, window.get_width(), window.get_height())
        cmd_ctx.cmd_set_primitive_topology(graphics.PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)

        # Draw second triangle first (opaque, smaller)
        cmd_ctx.cmd_draw_indexed(3, 1, 0, 3, 0)  # Start from vertex 3, draw 3 vertices

        # Draw first triangle second (semi-transparent, larger)
        cmd_ctx.cmd_draw_indexed(3, 1, 0, 0, 0)  # Start from vertex 0, draw 3 vertices

        cmd_ctx.cmd_end_rendering()
        cmd_ctx.cmd_present(window, color_image)
        core.submit_command_context(cmd_ctx)
        window.poll_events()

    print(f"\n=== Blend Demo Complete ===")

if __name__ == '__main__':
    main()
