import math

from long_march.grassland import graphics
from long_march.snow_mount import visualizer
import scipy
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
        self.rot_speed = 0.003

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
        if self.window.get_key(glfw.KEY_LEFT_CONTROL):
            move_signal[1] -= 1
        move_signal = np.asarray(move_signal)
        move_signal = move_signal * self.move_speed * period

        if self.window.get_mouse_button(glfw.MOUSE_BUTTON_LEFT):
            self.camera_orientation[1] -= cursor_diff[0] * self.rot_speed
            self.camera_orientation[0] -= cursor_diff[1] * self.rot_speed
            self.camera_orientation[0] = np.clip(self.camera_orientation[0], -math.pi / 2, math.pi / 2)
            self.camera_orientation[1] = self.camera_orientation[1] % (2 * math.pi)
        R = np.identity(4)

        R[:3, :3] = scipy.spatial.transform.Rotation.from_rotvec(
            [0, self.camera_orientation[1], 0]).as_matrix() @ scipy.spatial.transform.Rotation.from_rotvec(
            [self.camera_orientation[0], 0, 0]).as_matrix() @ scipy.spatial.transform.Rotation.from_rotvec(
            [0, 0, self.camera_orientation[2]]).as_matrix()

        self.camera_position += R[:3, :3] @ move_signal

        R[:3, 3] = self.camera_position

        # inverse matrix R
        R = np.linalg.inv(R)

        if effective:
            self.camera.view = R
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
    camera.proj = visualizer.perspective(fovy=math.radians(60), aspect=800 / 600, near=0.1, far=100)

    cam_controller = CameraController(window, camera)

    entity = vis_core.create_entity_mesh_object()
    entity.set_mesh(mesh)
    entity.set_material(visualizer.Material())
    scene = vis_core.create_scene()
    scene.add_entity(entity)

    entities = [entity]
    del entity

    last_frame_time = time.time()

    def resize_callback(w, h):
        nonlocal core, film, camera
        core.wait_gpu()
        film = vis_core.create_film(w, h)
        camera.proj = visualizer.perspective(fovy=math.radians(60), aspect=w / h, near=0.1, far=100)


    window.reg_resize_callback(resize_callback)

    while not window.should_close():
        current_frame_time = time.time()
        period = current_frame_time - last_frame_time
        context = core.create_command_context()
        vis_core.render(context, scene, camera, film)
        if len(entities):
            entities = []
        else:
            entity = vis_core.create_entity_mesh_object()
            entity.set_mesh(mesh)
            entity.set_material(visualizer.Material([0.8, 0.5, 0.5, 1.0]))
            entities = [entity]
            scene.add_entity(entities[0])
        context.cmd_present(window, film.get_image())
        context.submit()
        graphics.glfw_poll_events()
        last_frame_time = current_frame_time
        cam_controller.update(True)

    core.wait_gpu()


if __name__ == "__main__":
    main()

# print current working directory
