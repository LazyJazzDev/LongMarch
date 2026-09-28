import sys

import numpy as np
from long_march import graphics

import graphics_hello as gh

TRIANGLE_VERTICES = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]], np.float32)
TRIANGLE_INDICES = np.array([0, 1, 2], np.uint32)


def create_triangle_buffers(core, buffer_type=graphics.BUFFER_TYPE_DYNAMIC):
    vertex_buffer = core.create_buffer(TRIANGLE_VERTICES.nbytes, buffer_type)
    index_buffer = core.create_buffer(TRIANGLE_INDICES.nbytes, buffer_type)
    vertex_buffer.upload_data(TRIANGLE_VERTICES.tobytes())
    index_buffer.upload_data(TRIANGLE_INDICES.tobytes())
    return vertex_buffer, index_buffer


def require_ray_tracing_pipelines(api, core_message):
    if api == graphics.BACKEND_API_METAL:
        raise RuntimeError("Metal has no ray tracing pipelines; use --module ray_query")
    core = gh.initialize_graphics_hello(api, True)
    if not core.ray_tracing_support():
        raise RuntimeError(core_message)
    return core


class Module(gh.Module):
    def __init__(self, api=graphics.BACKEND_API_DEFAULT):
        super().__init__()
        self.core = require_ray_tracing_pipelines(api,
                                                  "Ray tracing pipelines are unavailable; use --module ray_query on Metal")

    def on_init(self):
        self.alive = True
        self.window = self.core.create_window(1280, 720, gh.graphics_hello_title(self.core.api()) +
                                              " Graphics Hello Ray Tracing")

        self.vertex_buffer, self.index_buffer = create_triangle_buffers(self.core)

        self.camera_object_buffer = self.core.create_buffer(2 * 64, graphics.BUFFER_TYPE_DYNAMIC)
        self.camera_object_buffer.upload_data(gh.camera_object(self.window, (0.0, 1.0, 5.0)))

        self.color_image = self.core.create_image(self.window.get_width(), self.window.get_height(),
                                                  graphics.IMAGE_FORMAT_R32G32B32A32_SFLOAT)

        shader = gh.load_shader("modules/raytracing/shaders/shader.slang")
        self.raygen_shader = self.core.create_shader(shader, "RayGenMain", "lib_6_3")
        self.miss_shader = self.core.create_shader(shader, "MissMain", "lib_6_3")
        self.closest_hit_shader = self.core.create_shader(shader, "ClosestHitMain", "lib_6_3")
        gh.log_info("Shader compiled successfully")

        self.blas = self.core.create_blas(self.vertex_buffer, self.index_buffer, 12)
        self.tlas = self.core.create_tlas([self.blas.make_instance(gh.instance_transform(np.identity(4)), 0, 0xFF, 0)])

        self.program = self.core.create_raytracing_program(self.raygen_shader, self.miss_shader,
                                                           self.closest_hit_shader)
        self.program.add_resource_binding(graphics.RESOURCE_TYPE_ACCELERATION_STRUCTURE, 1)
        self.program.add_resource_binding(graphics.RESOURCE_TYPE_WRITABLE_IMAGE, 1)
        self.program.add_resource_binding(graphics.RESOURCE_TYPE_UNIFORM_BUFFER, 1)
        self.program.finalize()

    def on_close(self):
        self.core.wait_gpu()
        del self.program, self.raygen_shader, self.miss_shader, self.closest_hit_shader
        del self.blas, self.tlas
        del self.color_image, self.camera_object_buffer, self.index_buffer, self.vertex_buffer

    def on_update(self):
        if self.update_alive():
            theta = self.rotation_angle()
            self.tlas.update_instances(
                [self.blas.make_instance(gh.instance_transform(gh.rotate_y(theta)), 0, 0xFF, 0)])

    def on_render(self):
        command_context = self.core.create_command_context()
        command_context.cmd_clear_image(self.color_image, [0.6, 0.7, 0.8, 1.0])
        command_context.cmd_bind_raytracing_program(self.program)
        command_context.cmd_bind_resources(0, self.tlas, graphics.BIND_POINT_RAYTRACING)
        command_context.cmd_bind_resources(1, [self.color_image], graphics.BIND_POINT_RAYTRACING)
        command_context.cmd_bind_resources(2, [self.camera_object_buffer], graphics.BIND_POINT_RAYTRACING)
        command_context.cmd_dispatch_rays(self.window.get_width(), self.window.get_height(), 1)
        command_context.cmd_present(self.window, self.color_image)
        self.core.submit_command_context(command_context)


if __name__ == "__main__":
    sys.exit(gh.main(module="raytracing"))
