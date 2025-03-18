import math

from long_march.grassland import graphics
from long_march.snow_mount import visualizer
import glfw
import time
import numpy as np


class CameraController:
    def __init__(self, window: graphics.Window, camera: visualizer.Camera):
        self.window = window
        self.camera = camera
        self.camera_position = np.asarray([0., 1., 5.])
        self.camera_orientation = np.asarray([0., 0., 0.])
        self.last_update_time = time.time()
        self.last_cursor_pos = self.window.get_cursor_pos()
        self.move_speed = 1.0
        print(type(self.last_cursor_pos))

    def update(self, effective=True):
        this_update_time = time.time()
        period = this_update_time - self.last_update_time
        this_cursor_pos = self.window.get_cursor_pos()
        cursor_diff = [this_cursor_pos[0] - self.last_cursor_pos[0], this_cursor_pos[1] - self.last_cursor_pos[1]]
        move_signal = [0., 0., 0.]
        if self.window.get_key(glfw.KEY_A):
            move_signal[0] += -1
        if self.window.get_key(glfw.KEY_D):
            move_signal[0] += 1
        if self.window.get_key(glfw.KEY_W):
            move_signal[2] += -1
        if self.window.get_key(glfw.KEY_S):
            move_signal[2] += 1
        if self.window.get_key(glfw.KEY_SPACE):
            move_signal[1] += 1
        if self.window.get_key(glfw.KEY_LEFT_SHIFT):
            move_signal[1] -= 1
        move_signal = np.asarray(move_signal)
        move_signal = move_signal * self.move_speed * period
        self.camera_position += move_signal
        if effective:
            self.camera.view = visualizer.look_at(self.camera_position, [0, 0, 0], [0, 1, 0])
        self.last_update_time = this_update_time
        self.last_cursor_pos = this_cursor_pos

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

    camera = vis_core.create_camera()
    camera.proj = visualizer.perspective(fovy=math.radians(90), aspect=800 / 600, near=0.1, far=100)
    camera.view = visualizer.look_at(eye=[0, 1, 5], center=[0, 0, 0], up=[0, 1, 0])

    cam_controller = CameraController(window, camera)

    entity = vis_core.create_entity_mesh_object()
    entity.set_mesh(mesh)
    entity.set_material(visualizer.Material())
    scene = vis_core.create_scene()
    scene.add_entity(entity)

    last_frame_time = time.time()

    while not window.should_close():
        current_frame_time = time.time()
        period = current_frame_time - last_frame_time
        context = core.create_command_context()
        # context.cmd_clear_image(film.get_image(), graphics.ColorClearValue(.6, .7, .8, 1.))
        vis_core.render(context, scene, camera, film)
        context.cmd_present(window, film.get_image())
        context.submit()
        graphics.glfw_poll_events()
        last_frame_time = current_frame_time
        cam_controller.update(True)

    core.wait_gpu()


if __name__ == "__main__":
    main()

# print current working directory
