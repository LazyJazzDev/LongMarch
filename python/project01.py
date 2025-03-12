import long_march
import long_march.grassland
import long_march.grassland.graphics as graphics
import glfw

def main():
    glfw.init()
    core_settings = graphics.CoreSettings()
    core_settings.frames_in_flight = 1
    core = graphics.create_core(settings=core_settings)
    print(core)
    window = core.create_window(resizable=True)
    window.set_title("Project01")

    color_frame = core.create_image(800, 600, graphics.IMAGE_FORMAT_R32G32B32A32_SFLOAT)
    print(color_frame)

    while not window.should_close():
        context = core.create_command_context()
        context.cmd_clear_image(color_frame, graphics.ColorClearValue(pow(1.8, 2.2), pow(2.1, 2.2), pow(2.4, 2.2), 1))
        context.cmd_present(window, color_frame)
        context.submit()
        graphics.glfw_poll_events()


if __name__ == "__main__":
    main()

# print current working directory
