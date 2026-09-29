import sys

from long_march import graphics

import graphics_hello as gh
from graphics_hello_cube import CUBE_INDICES, CUBE_VERTICES, cube_uniforms


class Module(gh.Module):
    def __init__(self, api=graphics.BACKEND_API_DEFAULT):
        super().__init__()
        self.core = gh.initialize_graphics_hello(api)

    def create_targets(self, width, height):
        self.color_image = self.core.create_image(width, height, graphics.IMAGE_FORMAT_R32G32B32A32_SFLOAT)
        self.depth_image = self.core.create_image(width, height, graphics.IMAGE_FORMAT_D32_SFLOAT)

    def on_framebuffer_resize(self, width, height):
        if width <= 0 or height <= 0:
            return
        self.core.wait_gpu()
        self.color_image = self.depth_image = None
        self.create_targets(width, height)

    def on_init(self):
        self.alive = True
        self.window = self.core.create_window(1280, 720, gh.graphics_hello_title(self.core.api()) +
                                              " Graphics Hello Resize", False, True)

        self.vertex_buffer = self.core.create_buffer(CUBE_VERTICES.nbytes, graphics.BUFFER_TYPE_STATIC)
        self.index_buffer = self.core.create_buffer(CUBE_INDICES.nbytes, graphics.BUFFER_TYPE_STATIC)
        self.vertex_buffer.upload_data(CUBE_VERTICES.tobytes())
        self.index_buffer.upload_data(CUBE_INDICES.tobytes())

        self.uniform_buffer = self.core.create_buffer(3 * 64, graphics.BUFFER_TYPE_DYNAMIC)

        self.create_targets(*self.window.get_framebuffer_size())
        self.window.register_framebuffer_resize_event(self.on_framebuffer_resize)

        shader = gh.load_shader("modules/resize/shaders/shader.slang")
        self.vertex_shader = self.core.create_shader(shader, "VSMain", "vs_6_0")
        self.fragment_shader = self.core.create_shader(shader, "PSMain", "ps_6_0")
        gh.log_info("Shader compiled successfully")

        self.program = self.core.create_program([graphics.IMAGE_FORMAT_R32G32B32A32_SFLOAT],
                                                graphics.IMAGE_FORMAT_D32_SFLOAT)
        self.program.add_input_binding(24, False)
        self.program.add_input_attribute(0, graphics.INPUT_TYPE_FLOAT3, 0)
        self.program.add_input_attribute(0, graphics.INPUT_TYPE_FLOAT3, 12)
        self.program.add_resource_binding(graphics.RESOURCE_TYPE_UNIFORM_BUFFER, 1)
        self.program.set_cull_mode(graphics.CULL_MODE_NONE)
        self.program.bind_shader(self.vertex_shader, graphics.SHADER_TYPE_VERTEX)
        self.program.bind_shader(self.fragment_shader, graphics.SHADER_TYPE_PIXEL)
        self.program.finalize()

    def on_close(self):
        self.core.wait_gpu()
        del self.program, self.vertex_shader, self.fragment_shader, self.color_image, self.depth_image
        del self.index_buffer, self.vertex_buffer, self.uniform_buffer

    def on_update(self):
        if self.update_alive():
            extent = self.color_image.extent()
            self.uniform_buffer.upload_data(cube_uniforms(self.rotation_angle(), extent.width / extent.height))

    def on_render(self):
        command_context = self.core.create_command_context()
        command_context.cmd_clear_image(self.color_image, [0.6, 0.7, 0.8, 1.0])
        command_context.cmd_clear_image(self.depth_image, 1.0)
        command_context.cmd_begin_rendering([self.color_image], self.depth_image)
        command_context.cmd_bind_program(self.program)
        command_context.cmd_bind_vertex_buffers(0, [self.vertex_buffer], [0])
        command_context.cmd_bind_index_buffer(self.index_buffer, 0)
        command_context.cmd_bind_resources(0, [self.uniform_buffer])

        extent = self.color_image.extent()
        command_context.cmd_set_viewport(0, 0, extent.width, extent.height, 0.0, 1.0)
        command_context.cmd_set_scissor(0, 0, extent.width, extent.height)
        command_context.cmd_set_primitive_topology(graphics.PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        command_context.cmd_draw_indexed(36, 1, 0, 0, 0)
        command_context.cmd_end_rendering()
        command_context.cmd_present(self.window, self.color_image)
        self.core.submit_command_context(command_context)


if __name__ == "__main__":
    sys.exit(gh.main(module="resize"))
