import struct
import sys

import numpy as np
from long_march import graphics

import graphics_hello as gh


def write_bmp(path, width, height, rgba):
    """32-bit BMP with alpha, bottom-up rows, as written by stbi_write_bmp."""
    pixels = np.frombuffer(rgba, np.uint8).reshape(height, width, 4)[::-1, :, [2, 1, 0, 3]]
    header_size = 14 + 108
    with open(path, "wb") as f:
        f.write(struct.pack("<2sIHHI", b"BM", header_size + pixels.nbytes, 0, 0, header_size))
        f.write(struct.pack("<IiiHHIIiiII", 108, width, height, 1, 32, 3, pixels.nbytes, 0, 0, 0, 0))
        f.write(struct.pack("<IIII", 0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000))
        f.write(b"sRGB" + bytes(48))
        f.write(pixels.tobytes())


class Module(gh.Module):
    def __init__(self, api=graphics.BACKEND_API_DEFAULT):
        super().__init__()
        self.core = gh.initialize_graphics_hello(api)
        self.first_frame = True

    def on_init(self):
        self.alive = True
        self.window = self.core.create_window(1280, 720, gh.graphics_hello_title(self.core.api()) +
                                              " Graphics Hello Triangle")

        vertices = np.array([
            [0.0, 0.5, 0.0, 1.0, 0.0, 0.0],
            [-0.5, -0.5, 0.0, 0.0, 0.0, 1.0],
            [0.5, -0.5, 0.0, 0.0, 1.0, 0.0],
        ], np.float32)
        indices = np.array([0, 1, 2], np.uint32)

        self.vertex_buffer = self.core.create_buffer(vertices.nbytes, graphics.BUFFER_TYPE_DYNAMIC)
        self.index_buffer = self.core.create_buffer(indices.nbytes, graphics.BUFFER_TYPE_DYNAMIC)
        self.vertex_buffer.upload_data(vertices.tobytes())
        self.index_buffer.upload_data(indices.tobytes())

        self.color_image = self.core.create_image(1280, 720, graphics.IMAGE_FORMAT_R8G8B8A8_UNORM)

        shader = gh.load_shader("modules/triangle/shaders/shader.slang")
        self.vertex_shader = self.core.create_shader(shader, "VSMain", "vs_6_0")
        self.fragment_shader = self.core.create_shader(shader, "PSMain", "ps_6_0")
        gh.log_info("Shader compiled successfully")

        self.program = self.core.create_program([self.color_image.format()], graphics.IMAGE_FORMAT_UNDEFINED)
        self.program.add_input_binding(24, False)
        self.program.add_input_attribute(0, graphics.INPUT_TYPE_FLOAT3, 0)
        self.program.add_input_attribute(0, graphics.INPUT_TYPE_FLOAT3, 12)
        self.program.bind_shader(self.vertex_shader, graphics.SHADER_TYPE_VERTEX)
        self.program.bind_shader(self.fragment_shader, graphics.SHADER_TYPE_PIXEL)
        self.program.finalize()

    def on_close(self):
        self.core.wait_gpu()
        del self.program, self.vertex_shader, self.fragment_shader, self.color_image
        del self.index_buffer, self.vertex_buffer

    def on_update(self):
        self.update_alive()

    def on_render(self):
        command_context = self.core.create_command_context()
        command_context.cmd_clear_image(self.color_image, [0.6, 0.7, 0.8, 1.0])
        command_context.cmd_begin_rendering([self.color_image], None)
        command_context.cmd_bind_program(self.program)
        command_context.cmd_bind_vertex_buffers(0, [self.vertex_buffer], [0])
        command_context.cmd_bind_index_buffer(self.index_buffer, 0)
        command_context.cmd_set_viewport(0, 0, 1280, 720, 0.0, 1.0)
        command_context.cmd_set_scissor(0, 0, 1280, 720)
        command_context.cmd_set_primitive_topology(graphics.PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        command_context.cmd_draw_indexed(3, 1, 0, 0, 0)
        command_context.cmd_end_rendering()
        command_context.cmd_present(self.window, self.color_image)
        self.core.submit_command_context(command_context)
        if self.first_frame:
            self.first_frame = False
            extent = self.color_image.extent()
            write_bmp(f"{gh.backend_name(self.core.api())}_hello_triangle.bmp", extent.width, extent.height,
                      self.color_image.download_data())


if __name__ == "__main__":
    sys.exit(gh.main(module="triangle"))
