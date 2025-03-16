from long_march.grassland import graphics
from long_march.snow_mount import visualizer
import glfw
import time
import open3d as o3d


def main():
    glfw.init()
    core_settings = graphics.CoreSettings()
    core_settings.frames_in_flight = 1
    core = graphics.Core(settings=core_settings)
    print(core)
    window = core.create_window(resizable=True)
    window.set_title("Project01")

    vis_core = visualizer.Core(core)
    mesh = vis_core.create_mesh()
    film = vis_core.create_film(800, 600)
    print(film)
    print(film.get_image(visualizer.FILM_CHANNEL_EXPOSURE))
    print(film.get_image(visualizer.FILM_CHANNEL_DEPTH))
    print(visualizer.FilmChannel(2))

    mesh_vertices = [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1]]
    mesh_indices = [0, 2, 1,
                    0, 3, 2,
                    1, 6, 5,
                    1, 2, 6,
                    5, 7, 4,
                    5, 6, 7,
                    4, 3, 0,
                    4, 7, 3,
                    3, 6, 2,
                    3, 7, 6,
                    0, 5, 4,
                    0, 1, 5]

    mesh.set_vertices(mesh_vertices)
    mesh.set_indices(mesh_indices)

    while not window.should_close():
        context = core.create_command_context()
        context.cmd_clear_image(film.get_image(), graphics.ColorClearValue(.6, .7, .8, 1.))
        context.cmd_present(window, film.get_image())
        context.submit()
        graphics.glfw_poll_events()

    core.wait_gpu()


if __name__ == "__main__":
    main()

# print current working directory
