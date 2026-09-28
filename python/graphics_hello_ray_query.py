import sys

from long_march import graphics

import graphics_hello as gh
from graphics_hello_raytracing import create_triangle_buffers
from graphics_hello_rt_multi_shader_group import UNIT_AABB, scene_instances


class Module(gh.Module):
    def __init__(self, api=graphics.BACKEND_API_DEFAULT):
        super().__init__()
        self.core = gh.initialize_graphics_hello(api)
        if not self.core.ray_query_support():
            raise RuntimeError("Ray queries are unavailable on the selected device/backend")

    def on_init(self):
        self.alive = True
        self.window = self.core.create_window(1280, 720, gh.graphics_hello_title(self.core.api()) +
                                              " Graphics Hello Ray Query")

        self.vertex_buffer, self.index_buffer = create_triangle_buffers(self.core)

        self.camera_object_buffer = self.core.create_buffer(2 * 64, graphics.BUFFER_TYPE_DYNAMIC)
        self.camera_object_buffer.upload_data(gh.camera_object(self.window, (0.0, 0.0, 5.0)))

        self.color_image = self.core.create_image(self.window.get_width(), self.window.get_height(),
                                                  graphics.IMAGE_FORMAT_R32G32B32A32_SFLOAT)

        self.compute_shader = self.core.create_shader(gh.load_shader("modules/ray_query/shaders/shader.slang"),
                                                      "CSMain", "cs_6_5")

        self.triangle_blas = self.core.create_blas(self.vertex_buffer, self.index_buffer, 12)
        aabb_buffer = self.core.create_buffer(UNIT_AABB.nbytes, graphics.BUFFER_TYPE_STATIC)
        aabb_buffer.upload_data(UNIT_AABB.tobytes())
        self.sphere_blas = self.core.create_blas(graphics.BufferRange(aabb_buffer), UNIT_AABB.nbytes, 1,
                                                 graphics.RAYTRACING_GEOMETRY_FLAG_OPAQUE)
        self.tlas = self.core.create_tlas(scene_instances(self.triangle_blas, self.sphere_blas))

        self.program = self.core.create_compute_program(self.compute_shader)
        self.program.add_resource_binding(graphics.RESOURCE_TYPE_ACCELERATION_STRUCTURE, 1)
        self.program.add_resource_binding(graphics.RESOURCE_TYPE_WRITABLE_IMAGE, 1)
        self.program.add_resource_binding(graphics.RESOURCE_TYPE_UNIFORM_BUFFER, 1)
        self.program.finalize()

    def on_close(self):
        self.core.wait_gpu()
        del self.program, self.compute_shader
        del self.tlas, self.sphere_blas, self.triangle_blas
        del self.color_image, self.camera_object_buffer, self.index_buffer, self.vertex_buffer

    def on_update(self):
        if self.update_alive():
            self.tlas.update_instances(scene_instances(self.triangle_blas, self.sphere_blas, self.rotation_angle()))

    def on_render(self):
        command_context = self.core.create_command_context()
        command_context.cmd_clear_image(self.color_image, [0.6, 0.7, 0.8, 1.0])
        command_context.cmd_bind_compute_program(self.program)
        command_context.cmd_bind_resources(0, self.tlas, graphics.BIND_POINT_COMPUTE)
        command_context.cmd_bind_resources(1, [self.color_image], graphics.BIND_POINT_COMPUTE)
        command_context.cmd_bind_resources(2, [self.camera_object_buffer], graphics.BIND_POINT_COMPUTE)
        extent = self.color_image.extent()
        command_context.cmd_dispatch((extent.width + 7) // 8, (extent.height + 7) // 8, 1)
        command_context.cmd_present(self.window, self.color_image)
        self.core.submit_command_context(command_context)


if __name__ == "__main__":
    sys.exit(gh.main(module="ray_query"))
