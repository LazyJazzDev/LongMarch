import pathlib

from long_march import graphics
import numpy as np
import sys

def main():
    print("=== LongMarch Graphics Hello Texture Demo ===")
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
    window = core.create_window(1280, 720, "[Python] Graphics Hello Texture", resizable=True)
    print(f"Window Created: {window}")

    # Load shader file
    shader_path = "./shaders/texture_shader.hlsl"
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
    program = core.create_program([graphics.IMAGE_FORMAT_R32G32B32A32_SFLOAT], graphics.IMAGE_FORMAT_D32_SFLOAT)
    program.add_input_binding(20)  # sizeof(Vertex) = 5 floats * 4 bytes = 20 bytes
    program.add_input_attribute(0, graphics.INPUT_TYPE_FLOAT3, 0)  # position
    program.add_input_attribute(0, graphics.INPUT_TYPE_FLOAT2, 12)  # tex_coord
    program.add_resource_binding(graphics.RESOURCE_TYPE_IMAGE, 1)
    program.add_resource_binding(graphics.RESOURCE_TYPE_SAMPLER, 1)
    program.bind_shader(vertex_shader, graphics.SHADER_TYPE_VERTEX)
    program.bind_shader(pixel_shader, graphics.SHADER_TYPE_PIXEL)
    program.finalize()
    print(f"Program Created: {program}")

    print("Preparing triangle geometry with texture coordinates...")
    # Triangle vertices with position and texture coordinates
    vertices = np.array([
        [0.0, 0.5, 0.0, 0.5, 0.0],    # Top vertex
        [-0.5, -0.5, 0.0, 0.0, 1.0],  # Bottom left
        [0.5, -0.5, 0.0, 1.0, 1.0],   # Bottom right
    ], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.uint32)
    vertices_bytes = vertices.tobytes()
    indices_bytes = indices.tobytes()

    print(f"Vertex data: {len(vertices_bytes)} bytes ({len(vertices)} vertices)")
    print(f"Index data: {len(indices_bytes)} bytes ({len(indices)} indices)")

    print("Creating buffers...")
    vertex_buffer = core.create_buffer(len(vertices_bytes), graphics.BUFFER_TYPE_STATIC)
    vertex_buffer.upload_data(vertices_bytes)
    index_buffer = core.create_buffer(len(indices_bytes), graphics.BUFFER_TYPE_STATIC)
    index_buffer.upload_data(indices_bytes)
    print(f"Vertex Buffer: {vertex_buffer}")
    print(f"Index Buffer: {index_buffer}")

    print("Creating frame buffers...")
    color_image = core.create_image(window.get_width(), window.get_height(), graphics.IMAGE_FORMAT_R32G32B32A32_SFLOAT)
    depth_image = core.create_image(window.get_width(), window.get_height(), graphics.IMAGE_FORMAT_D32_SFLOAT)
    print(f"Color Image: {color_image}")
    print(f"Depth Image: {depth_image}")

    print("Creating texture...")
    texture_size = 256
    texture_image = core.create_image(texture_size, texture_size, graphics.IMAGE_FORMAT_R8G8B8A8_UNORM)

    # Generate checkerboard pattern texture
    texture_data = np.zeros((texture_size, texture_size), dtype=np.uint32)
    for i in range(texture_size):
        for j in range(texture_size):
            pixel = i ^ j  # XOR pattern
            pixel = pixel | (pixel << 8) | (pixel << 16) | 0xFF000000  # RGBA format
            texture_data[i, j] = pixel

    texture_image.upload_data(texture_data.flatten().tobytes())
    print(f"Texture Image: {texture_image}")

    print("Creating sampler...")
    sampler_info = graphics.SamplerInfo(graphics.FILTER_MODE_LINEAR)
    sampler = core.create_sampler(sampler_info)
    print(f"Sampler: {sampler}")

    def on_resize(width, height):
        print(f"Window resized to {width}x{height} - Recreating frame buffers")
        nonlocal color_image, depth_image
        color_image = core.create_image(window.get_width(), window.get_height(), graphics.IMAGE_FORMAT_R32G32B32A32_SFLOAT)
        depth_image = core.create_image(window.get_width(), window.get_height(), graphics.IMAGE_FORMAT_D32_SFLOAT)
        print(f"New Color Image: {color_image}")
        print(f"New Depth Image: {depth_image}")

    window.register_resize_event(on_resize)

    print("\n=== Starting Render Loop ===")

    while not window.should_close():
        # Render frame
        cmd_ctx = core.create_command_context()
        cmd_ctx.cmd_clear_image(color_image, [0.6, 0.7, 0.8, 1.0])  # Light blue background
        cmd_ctx.cmd_clear_image(depth_image, 1.0)  # Clear depth to 1.0
        cmd_ctx.cmd_begin_rendering([color_image], depth_image)
        cmd_ctx.cmd_bind_program(program)
        cmd_ctx.cmd_bind_vertex_buffers(0, [vertex_buffer], [0])
        cmd_ctx.cmd_bind_index_buffer(index_buffer, 0)
        cmd_ctx.cmd_bind_resources(0, [texture_image])
        cmd_ctx.cmd_bind_resources(1, [sampler])
        cmd_ctx.cmd_set_viewport(0, 0, window.get_width(), window.get_height(), 0.0, 1.0)
        cmd_ctx.cmd_set_scissor(0, 0, window.get_width(), window.get_height())
        cmd_ctx.cmd_set_primitive_topology(graphics.PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        cmd_ctx.cmd_draw_indexed(3, 1, 0, 0, 0)
        cmd_ctx.cmd_end_rendering()
        cmd_ctx.cmd_present(window, color_image)
        core.submit_command_context(cmd_ctx)
        window.poll_events()

    print(f"\n=== Texture Demo Complete ===")

if __name__ == '__main__':
    main()
