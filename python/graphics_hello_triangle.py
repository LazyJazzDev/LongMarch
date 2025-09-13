from long_march import graphics
import numpy as np

def main():
    core_settings=graphics.CoreSettings()
    print(core_settings)

    core=graphics.Core()
    print(core)
    core.init_auto()
    print(core)
    window=core.create_window(800, 600, "Hello Triangle", resizable=True)

    # load file shaders/hello_triangle.hlsl into a str
    with open("shaders/hello_triangle.hlsl", "r") as f:
        shader_code = f.read()
    vertex_shader=core.create_shader(shader_code, "VSMain", "vs_6_0")
    pixel_shader=core.create_shader(shader_code, "PSMain", "ps_6_0")
    program=core.create_program([graphics.IMAGE_FORMAT_R8G8B8A8_UNORM])
    program.bind_shader(vertex_shader, graphics.SHADER_TYPE_VERTEX)
    program.bind_shader(pixel_shader, graphics.SHADER_TYPE_PIXEL)
    program.add_input_binding(24)
    program.add_input_attribute(0, graphics.INPUT_TYPE_FLOAT3, 0)
    program.add_input_attribute(0, graphics.INPUT_TYPE_FLOAT3, 12)
    program.finalize()

    vertices = np.array([
        [0.0, 0.5, 0.0, 1.0, 0.0, 0.0],
        [-0.5, -0.5, 0.0, 0.0, 0.0, 1.0],
        [0.5, -0.5, 0.0, 0.0, 1.0, 0.0],
    ], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.uint32)
    vertices_bytes = vertices.tobytes()
    indices_bytes = indices.tobytes()

    print(len(vertices_bytes), type(vertices_bytes), len(indices_bytes), type(indices_bytes))
    vertex_buffer=core.create_buffer(len(vertices_bytes), graphics.BUFFER_TYPE_STATIC)
    vertex_buffer.upload_data(vertices_bytes)
    index_buffer=core.create_buffer(len(indices_bytes), graphics.BUFFER_TYPE_STATIC)
    index_buffer.upload_data(indices_bytes)

    frame_image=core.create_image(window.get_width(), window.get_height(), graphics.IMAGE_FORMAT_R8G8B8A8_UNORM)

    print(window)
    while not window.should_close():
        cmd_ctx=core.create_command_context()
        cmd_ctx.cmd_clear_image(frame_image, [0.6, 0.7, 0.9, 1.0])
        cmd_ctx.cmd_bind_program(program)
        cmd_ctx.cmd_begin_rendering([frame_image])
        cmd_ctx.cmd_set_viewport(0, 0, window.get_width(), window.get_height())
        cmd_ctx.cmd_set_scissor(0, 0, window.get_width(), window.get_height())
        cmd_ctx.cmd_set_primitive_topology(graphics.PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        cmd_ctx.cmd_bind_vertex_buffers(0, [vertex_buffer])
        cmd_ctx.cmd_bind_index_buffer(index_buffer)
        cmd_ctx.cmd_draw_indexed(3, 1, 0, 0, 0)
        cmd_ctx.cmd_end_rendering()
        cmd_ctx.cmd_present(window, frame_image)
        core.submit_command_context(cmd_ctx)
        window.poll_events()

if __name__ == '__main__':
    main()
