import sys

from long_march import graphics

import graphics_hello as gh


class Module(gh.Module):
    def __init__(self, api=graphics.BACKEND_API_DEFAULT):
        super().__init__()
        self.core = gh.initialize_graphics_hello(api)

    def on_init(self):
        self.alive = True
        self.window = self.core.create_window(1280, 720, gh.graphics_hello_title(self.core.api()) +
                                              " Graphics Hello SDR Sample")

        self.color_image = self.core.create_image(1280, 720, graphics.IMAGE_FORMAT_R32G32B32A32_SFLOAT)

        shader = gh.load_shader("modules/sdr_sample/shaders/shader.slang")
        self.vertex_shader = self.core.create_shader(shader, "VSMain", "vs_6_0")
        self.fragment_shader = self.core.create_shader(shader, "PSMain", "ps_6_0")
        gh.log_info("Shader compiled successfully")

        self.program = self.core.create_program([graphics.IMAGE_FORMAT_R32G32B32A32_SFLOAT],
                                                graphics.IMAGE_FORMAT_UNDEFINED)
        self.program.bind_shader(self.vertex_shader, graphics.SHADER_TYPE_VERTEX)
        self.program.bind_shader(self.fragment_shader, graphics.SHADER_TYPE_PIXEL)
        self.program.finalize()

    def on_close(self):
        self.core.wait_gpu()
        del self.program, self.vertex_shader, self.fragment_shader, self.color_image

    def on_update(self):
        self.update_alive()

    def on_render(self):
        command_context = self.core.create_command_context()
        command_context.cmd_clear_image(self.color_image, [0.6, 0.7, 0.8, 1.0])
        command_context.cmd_begin_rendering([self.color_image], None)
        command_context.cmd_bind_program(self.program)
        command_context.cmd_set_viewport(0, 0, 1280, 720, 0.0, 1.0)
        command_context.cmd_set_scissor(0, 0, 1280, 720)
        command_context.cmd_set_primitive_topology(graphics.PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        command_context.cmd_draw(6, 1, 0, 0)
        command_context.cmd_end_rendering()
        command_context.cmd_present(self.window, self.color_image)
        self.core.submit_command_context(command_context)


if __name__ == "__main__":
    sys.exit(gh.main(module="sdr_sample"))
