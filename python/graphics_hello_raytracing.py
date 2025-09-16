import pathlib
import math

from long_march import graphics
import numpy as np
import sys

def main():
    print("=== LongMarch Graphics Hello Ray Tracing Demo ===")
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
    window = core.create_window(1280, 720, "[Python] Graphics Hello Ray Tracing", resizable=True)
    print(f"Window Created: {window}")

    # Load shader file
    shader_path = "./shaders/raytracing_shader.hlsl"
    print(f"Loading shader from: {shader_path}")
    print(f"Shader file exists: {pathlib.Path(shader_path).exists()}")

    with open(shader_path, "r") as f:
        shader_code = f.read()

    print("Compiling ray tracing shaders...")
    raygen_shader = core.create_shader(shader_code, "RayGenMain", "lib_6_3")
    miss_shader = core.create_shader(shader_code, "MissMain", "lib_6_3")
    closest_hit_shader = core.create_shader(shader_code, "ClosestHitMain", "lib_6_3")
    print(f"Ray Generation Shader: {raygen_shader}")
    print(f"Miss Shader: {miss_shader}")
    print(f"Closest Hit Shader: {closest_hit_shader}")

    print("Creating ray tracing program...")
    program = core.create_raytracing_program(raygen_shader, miss_shader, closest_hit_shader)
    program.add_resource_binding(graphics.RESOURCE_TYPE_ACCELERATION_STRUCTURE, 1)
    program.add_resource_binding(graphics.RESOURCE_TYPE_WRITABLE_IMAGE, 1)
    program.add_resource_binding(graphics.RESOURCE_TYPE_UNIFORM_BUFFER, 1)
    program.finalize()
    print(f"Ray Tracing Program: {program}")

    print("Preparing triangle geometry...")
    # Simple triangle vertices
    vertices = np.array([
        [-1.0, -1.0, 0.0],  # Bottom left
        [1.0, -1.0, 0.0],   # Bottom right
        [0.0, 1.0, 0.0],    # Top center
    ], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.uint32)
    vertices_bytes = vertices.tobytes()
    indices_bytes = indices.tobytes()

    print(f"Vertex data: {len(vertices_bytes)} bytes ({len(vertices)} vertices)")
    print(f"Index data: {len(indices_bytes)} bytes ({len(indices)} indices)")

    print("Creating geometry buffers...")
    vertex_buffer = core.create_buffer(len(vertices_bytes), graphics.BUFFER_TYPE_DYNAMIC)
    vertex_buffer.upload_data(vertices_bytes)
    index_buffer = core.create_buffer(len(indices_bytes), graphics.BUFFER_TYPE_DYNAMIC)
    index_buffer.upload_data(indices_bytes)
    print(f"Vertex Buffer: {vertex_buffer}")
    print(f"Index Buffer: {index_buffer}")

    print("Creating camera buffer...")
    # Camera setup
    aspect_ratio = window.get_width() / window.get_height()
    fov = math.radians(60.0)
    near_plane = 0.1
    far_plane = 10.0

    # Create perspective projection matrix
    f = 1.0 / math.tan(fov / 2.0)
    proj_matrix = np.array([
        [f / aspect_ratio, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far_plane + near_plane) / (near_plane - far_plane), -1],
        [0, 0, (2 * far_plane * near_plane) / (near_plane - far_plane), 0]
    ], dtype=np.float32)

    # Create view matrix (camera at (0, 1, 5) looking at origin)
    eye = np.array([0.0, 1.0, 5.0], dtype=np.float32)
    target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

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

    # Camera object structure
    camera_object = {
        'screen_to_camera': np.linalg.inv(proj_matrix),
        'camera_to_world': np.linalg.inv(view_matrix)
    }

    camera_buffer = core.create_buffer(128, graphics.BUFFER_TYPE_DYNAMIC)  # 2 * 4x4 matrices
    camera_data = np.concatenate([
        camera_object['screen_to_camera'].flatten(),
        camera_object['camera_to_world'].flatten()
    ]).astype(np.float32)
    camera_buffer.upload_data(camera_data.tobytes())
    print(f"Camera Buffer: {camera_buffer}")

    print("Creating acceleration structures...")
    blas = core.create_blas(vertex_buffer, index_buffer, 12)  # 12 bytes per vertex (3 floats)
    print(f"Bottom-Level Acceleration Structure: {blas}")

    # Create ray tracing instance
    instance = graphics.RayTracingInstance()
    instance.transform = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ]
    instance.instance_id = 0
    instance.instance_mask = 0xFF
    instance.instance_hit_group_offset = 0
    instance.instance_flags = graphics.RAYTRACING_INSTANCE_FLAG_NONE
    instance.acceleration_structure = blas

    tlas = core.create_tlas([instance])
    print(f"Top-Level Acceleration Structure: {tlas}")

    print("Creating output image...")
    color_image = core.create_image(window.get_width(), window.get_height(), graphics.IMAGE_FORMAT_R32G32B32A32_SFLOAT)
    print(f"Color Image: {color_image}")

    def on_resize(width, height):
        print(f"Window resized to {width}x{height} - Recreating resources")
        nonlocal color_image, camera_buffer, camera_data

        # Recreate color image
        color_image = core.create_image(window.get_width(), window.get_height(), graphics.IMAGE_FORMAT_R32G32B32A32_SFLOAT)

        # Update camera matrices for new aspect ratio
        aspect_ratio = width / height
        f = 1.0 / math.tan(fov / 2.0)
        proj_matrix = np.array([
            [f / aspect_ratio, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far_plane + near_plane) / (near_plane - far_plane), -1],
            [0, 0, (2 * far_plane * near_plane) / (near_plane - far_plane), 0]
        ], dtype=np.float32)

        camera_object['screen_to_camera'] = np.linalg.inv(proj_matrix)
        camera_data = np.concatenate([
            camera_object['screen_to_camera'].flatten(),
            camera_object['camera_to_world'].flatten()
        ]).astype(np.float32)
        camera_buffer.upload_data(camera_data.tobytes())

        print(f"New Color Image: {color_image}")

    window.register_resize_event(on_resize)

    print("\n=== Starting Ray Tracing Render Loop ===")
    theta = 0.0

    while not window.should_close():
        # Update rotation
        theta += math.radians(0.1)

        # Update TLAS with rotated triangle
        rotation_matrix = np.array([
            [math.cos(theta), 0, math.sin(theta), 0],
            [0, 1, 0, 0],
            [-math.sin(theta), 0, math.cos(theta), 0]
        ], dtype=np.float32)

        instance.transform = rotation_matrix.tolist()
        tlas.update_instances([instance])

        # Render frame
        cmd_ctx = core.create_command_context()
        cmd_ctx.cmd_clear_image(color_image, [0.6, 0.7, 0.8, 1.0])  # Light blue background
        cmd_ctx.cmd_bind_raytracing_program(program)
        cmd_ctx.cmd_bind_resources(0, tlas, graphics.BIND_POINT_RAYTRACING)
        cmd_ctx.cmd_bind_resources(1, [color_image], graphics.BIND_POINT_RAYTRACING)
        cmd_ctx.cmd_bind_resources(2, [camera_buffer], graphics.BIND_POINT_RAYTRACING)
        cmd_ctx.cmd_dispatch_rays(window.get_width(), window.get_height(), 1)
        cmd_ctx.cmd_present(window, color_image)
        core.submit_command_context(cmd_ctx)
        window.poll_events()

    print(f"\n=== Ray Tracing Demo Complete ===")

if __name__ == '__main__':
    main()
