import pathlib
import math

from long_march import graphics
import numpy as np
import sys

def main():
    print("=== LongMarch Graphics Hello Cube Demo ===")
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
    window = core.create_window(1280, 720, "[Python] Graphics Hello Cube", resizable=True)
    print(f"Window Created: {window}")

    # Load shader file
    shader_path = "./shaders/cube_shader.hlsl"
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
    program.add_input_binding(24)  # sizeof(Vertex) = 6 floats * 4 bytes = 24 bytes
    program.add_input_attribute(0, graphics.INPUT_TYPE_FLOAT3, 0)  # position
    program.add_input_attribute(0, graphics.INPUT_TYPE_FLOAT3, 12)  # color
    program.add_resource_binding(graphics.RESOURCE_TYPE_UNIFORM_BUFFER, 1)
    program.set_cull_mode(graphics.CULL_MODE_NONE)
    program.bind_shader(vertex_shader, graphics.SHADER_TYPE_VERTEX)
    program.bind_shader(pixel_shader, graphics.SHADER_TYPE_PIXEL)
    program.finalize()
    print(f"Program Created: {program}")

    print("Preparing cube geometry...")
    # Cube vertices with position and color
    vertices = np.array([
        # Front face
        [-1.0, -1.0,  1.0, 0.0, 0.0, 0.0],  # 0: black
        [ 1.0, -1.0,  1.0, 1.0, 0.0, 0.0],  # 1: red
        [-1.0,  1.0,  1.0, 0.0, 1.0, 0.0],  # 2: green
        [ 1.0,  1.0,  1.0, 1.0, 1.0, 0.0],  # 3: yellow
        # Back face
        [-1.0, -1.0, -1.0, 0.0, 0.0, 1.0],  # 4: blue
        [ 1.0, -1.0, -1.0, 1.0, 0.0, 1.0],  # 5: magenta
        [-1.0,  1.0, -1.0, 0.0, 1.0, 1.0],  # 6: cyan
        [ 1.0,  1.0, -1.0, 1.0, 1.0, 1.0],  # 7: white
    ], dtype=np.float32)

    # Cube indices (12 triangles)
    indices = np.array([
        0, 1, 2,  2, 1, 3,  # Front face
        2, 3, 6,  6, 3, 7,  # Top face
        6, 7, 4,  4, 7, 5,  # Back face
        4, 5, 0,  0, 5, 1,  # Bottom face
        1, 5, 3,  3, 5, 7,  # Right face
        0, 2, 4,  4, 2, 6,  # Left face
    ], dtype=np.uint32)

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

    print("Creating uniform buffer...")
    uniform_buffer = core.create_buffer(192, graphics.BUFFER_TYPE_DYNAMIC)  # 3 * 4x4 matrices
    print(f"Uniform Buffer: {uniform_buffer}")

    print("Creating frame buffers...")
    color_image = core.create_image(window.get_width(), window.get_height(), graphics.IMAGE_FORMAT_R32G32B32A32_SFLOAT)
    depth_image = core.create_image(window.get_width(), window.get_height(), graphics.IMAGE_FORMAT_D32_SFLOAT)
    print(f"Color Image: {color_image}")
    print(f"Depth Image: {depth_image}")

    def on_resize(width, height):
        print(f"Window resized to {width}x{height} - Recreating frame buffers")
        nonlocal color_image, depth_image
        color_image = core.create_image(window.get_width(), window.get_height(), graphics.IMAGE_FORMAT_R32G32B32A32_SFLOAT)
        depth_image = core.create_image(window.get_width(), window.get_height(), graphics.IMAGE_FORMAT_D32_SFLOAT)
        print(f"New Color Image: {color_image}")
        print(f"New Depth Image: {depth_image}")

    window.register_resize_event(on_resize)

    print("\n=== Starting Render Loop ===")
    rotation_angle = 0.0

    while not window.should_close():
        # Update rotation
        rotation_angle += math.radians(1.0)
        if rotation_angle > math.radians(360.0):
            rotation_angle -= math.radians(360.0)

        # Create transformation matrices
        # Model matrix (rotation around Y axis) - matching C++ glm::rotate (COLUMN MAJOR!)
        cos_r = math.cos(rotation_angle)
        sin_r = math.sin(rotation_angle)
        # glm::rotate around Y-axis produces this exact matrix (column-major)
        model_matrix = np.array([
            [cos_r, 0, -sin_r, 0],
            [0, 1, 0, 0],
            [sin_r, 0, cos_r, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # View matrix (camera at (0, 0, 5) looking at origin) - matching C++ glm::lookAt
        eye = np.array([0.0, 0.0, 5.0], dtype=np.float32)
        target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        # Calculate lookAt matrix (same as glm::lookAt)
        z_axis = eye - target
        z_axis = z_axis / np.linalg.norm(z_axis)
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        view_matrix = np.array([
            [x_axis[0], y_axis[0], z_axis[0], 0],
            [x_axis[1], y_axis[1], z_axis[1], 0],
            [x_axis[2], y_axis[2], z_axis[2], 0],
            [-np.dot(x_axis, eye), -np.dot(y_axis, eye), -np.dot(z_axis, eye), 1]
        ], dtype=np.float32)

        # Projection matrix (perspectiveZO - matching C++ glm::perspectiveZO)
        aspect_ratio = window.get_width() / window.get_height()
        fov = math.radians(45.0)
        near_plane = 3.5
        far_plane = 6.5

        f = 1.0 / math.tan(fov / 2.0)
        # glm::perspectiveZO uses 0-to-1 depth range (not -1-to-1)
        proj_matrix = np.array([
            [f / aspect_ratio, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, far_plane / (near_plane - far_plane), -1],
            [0, 0, (near_plane * far_plane) / (near_plane - far_plane), 0]
        ], dtype=np.float32)

        # Upload uniform data
        uniform_data = np.concatenate([
            model_matrix.flatten(),
            view_matrix.flatten(),
            proj_matrix.flatten()
        ]).astype(np.float32)
        uniform_buffer.upload_data(uniform_data.tobytes())

        # Render frame
        cmd_ctx = core.create_command_context()
        cmd_ctx.cmd_clear_image(color_image, [0.6, 0.7, 0.8, 1.0])  # Light blue background
        cmd_ctx.cmd_clear_image(depth_image, 1.0)  # Clear depth to 1.0
        cmd_ctx.cmd_begin_rendering([color_image], depth_image)
        cmd_ctx.cmd_bind_program(program)
        cmd_ctx.cmd_bind_vertex_buffers(0, [vertex_buffer], [0])
        cmd_ctx.cmd_bind_index_buffer(index_buffer, 0)
        cmd_ctx.cmd_bind_resources(0, [uniform_buffer])
        cmd_ctx.cmd_set_viewport(0, 0, window.get_width(), window.get_height(), 0.0, 1.0)
        cmd_ctx.cmd_set_scissor(0, 0, window.get_width(), window.get_height())
        cmd_ctx.cmd_set_primitive_topology(graphics.PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        cmd_ctx.cmd_draw_indexed(36, 1, 0, 0, 0)  # 12 triangles * 3 vertices = 36 indices
        cmd_ctx.cmd_end_rendering()
        cmd_ctx.cmd_present(window, color_image)
        core.submit_command_context(cmd_ctx)
        window.poll_events()


    print(f"\n=== Cube Demo Complete ===")

if __name__ == '__main__':
    main()
