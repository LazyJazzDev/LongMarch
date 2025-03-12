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

    while not window.should_close():
        if (window.get_key(glfw.KEY_A)):
            print('A')
        if (window.get_key(glfw.KEY_D)):
            print('D')
        if (window.get_key(glfw.KEY_W)):
            print('W')
        if (window.get_key(glfw.KEY_S)):
            print('S')
        graphics.glfw_poll_events()


if __name__ == "__main__":
    main()

# print current working directory
