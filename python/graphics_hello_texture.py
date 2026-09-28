import sys

import numpy as np
from long_march import graphics

import graphics_hello as gh


class Module(gh.Module):
    def __init__(self, api=graphics.BACKEND_API_DEFAULT):
        super().__init__()
        self.core = gh.initialize_graphics_hello(api)

    def on_init(self):
        self.alive = True
        self.window = self.core.create_window(1280, 720, gh.graphics_hello_title(self.core.api()) +
                                              " Graphics Hello Texture")

        vertices = np.array([
            [0.0, 0.5, 0.0, 0.5, 0.0],
            [-0.5, -0.5, 0.0, 0.0, 1.0],
            [0.5, -0.5, 0.0, 1.0, 1.0],
        ], np.float32)
        indices = np.array([0, 1, 2], np.uint32)

        self.vertex_buffer = self.core.create_buffer(vertices.nbytes, graphics.BUFFER_TYPE_STATIC)
        self.index_buffer = self.core.create_buffer(indices.nbytes, graphics.BUFFER_TYPE_STATIC)
        self.vertex_buffer.upload_data(vertices.tobytes())
        self.index_buffer.upload_data(indices.tobytes())

        self.color_image = self.core.create_image(1280, 720, graphics.IMAGE_FORMAT_R32G32B32A32_SFLOAT)
        self.depth_image = self.core.create_image(1280, 720, graphics.IMAGE_FORMAT_D32_SFLOAT)

        texture_side_length = 256
        self.texture_image = self.core.create_image(texture_side_length, texture_side_length,
                                                    graphics.IMAGE_FORMAT_R8G8B8A8_UNORM)
        self.sampler = self.core.create_sampler(graphics.SamplerInfo(graphics.FILTER_MODE_LINEAR))

        i, j = np.indices((texture_side_length, texture_side_length), np.uint32)
        pixel = i ^ j
        texture_data = pixel | (pixel << 8) | (pixel << 16) | np.uint32(0xFF000000)
        self.texture_image.upload_data(texture_data.astype(np.uint32).tobytes())

        shader = gh.load_shader("modules/texture/shaders/shader.slang")
        self.vertex_shader = self.core.create_shader(shader, "VSMain", "vs_6_0")
        self.fragment_shader = self.core.create_shader(shader, "PSMain", "ps_6_0")
        gh.log_info("Shader compiled successfully")

        self.program = self.core.create_program([graphics.IMAGE_FORMAT_R32G32B32A32_SFLOAT],
                                                graphics.IMAGE_FORMAT_D32_SFLOAT)
        self.program.add_input_binding(20, False)
        self.program.add_input_attribute(0, graphics.INPUT_TYPE_FLOAT3, 0)
        self.program.add_input_attribute(0, graphics.INPUT_TYPE_FLOAT2, 12)
        self.program.add_resource_binding(graphics.RESOURCE_TYPE_IMAGE, 1)
        self.program.add_resource_binding(graphics.RESOURCE_TYPE_SAMPLER, 1)
        self.program.bind_shader(self.vertex_shader, graphics.SHADER_TYPE_VERTEX)
        self.program.bind_shader(self.fragment_shader, graphics.SHADER_TYPE_PIXEL)
        self.program.finalize()

    def on_close(self):
        self.core.wait_gpu()
        del self.program, self.vertex_shader, self.fragment_shader, self.sampler, self.texture_image
        del self.color_image, self.depth_image, self.index_buffer, self.vertex_buffer

    def on_update(self):
        self.update_alive()

    def on_render(self):
        command_context = self.core.create_command_context()
        command_context.cmd_clear_image(self.color_image, [0.6, 0.7, 0.8, 1.0])
        command_context.cmd_clear_image(self.depth_image, 1.0)
        command_context.cmd_begin_rendering([self.color_image], self.depth_image)
        command_context.cmd_bind_program(self.program)
        command_context.cmd_bind_vertex_buffers(0, [self.vertex_buffer], [0])
        command_context.cmd_bind_index_buffer(self.index_buffer, 0)
        command_context.cmd_bind_resources(0, [self.texture_image])
        command_context.cmd_bind_resources(1, [self.sampler])
        command_context.cmd_set_viewport(0, 0, 1280, 720, 0.0, 1.0)
        command_context.cmd_set_scissor(0, 0, 1280, 720)
        command_context.cmd_set_primitive_topology(graphics.PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        command_context.cmd_draw_indexed(3, 1, 0, 0, 0)
        command_context.cmd_end_rendering()
        command_context.cmd_present(self.window, self.color_image)
        self.core.submit_command_context(command_context)


if __name__ == "__main__":
    sys.exit(gh.main(module="texture"))
