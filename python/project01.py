import math
import time
import scipy
import long_march
from long_march.grassland import graphics
from long_march.snow_mount import solver
from long_march.snow_mount import visualizer
import glfw
glfw.init()

import numpy as np


class Scene:
    def __init__(self, vis_core: visualizer.Core):
        self.vis_scene = vis_core.create_scene()
        self.sol_scene = solver.Scene()


class RigidObject:
    def __init__(self, scene: Scene, vertices, indices):
        self.scene = scene
        self.vertices = np.asarray(vertices)
        self.indices = np.asarray(indices)
        self.vis_mesh = scene.vis_scene.get_core().create_mesh()
        self.vis_mesh.set_vertices(vertices)
        self.vis_mesh.set_indices(indices)
        mesh_sdf = long_march.grassland.math.MeshSDF(vertices, indices)
        self.sol_rigid_object = solver.RigidObject(mesh_sdf)
        self.sol_rigid_id = scene.sol_scene.add_rigid_object(self.sol_rigid_object)
        self.vis_entity = scene.vis_scene.get_core().create_entity_mesh_object()
        self.vis_entity.set_mesh(self.vis_mesh)
        self.vis_entity.set_material(visualizer.Material([0.8, 0.8, 0.8, 1.0]))
        scene.vis_scene.add_entity(self.vis_entity)

    def set_transform(self, R, t):
        transform4x4 = np.identity(4)
        transform4x4[:3, :3] = R
        transform4x4[:3, 3] = t
        self.vis_entity.set_transform(transform4x4)

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
    core_settings = graphics.CoreSettings()
    core_settings.frames_in_flight = 1

    graphics_core = graphics.Core(graphics.BACKEND_API_VULKAN, core_settings)
    vis_core = visualizer.Core(graphics_core)

    window = graphics_core.create_window(1280, 720, "Project01")

    camera = vis_core.create_camera(proj=visualizer.perspective(math.radians(60), 1280 / 720, 0.1, 100.0))
    camera_controller = CameraController(window, camera)

    film = vis_core.create_film(1280, 720)

    scene = Scene(vis_core)

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

    rigid_object = RigidObject(scene, mesh_vertices, mesh_indices)

    while not window.should_close():
        camera_controller.update(True)
        context = graphics_core.create_command_context()
        vis_core.render(context, scene.vis_scene, camera, film)
        context.cmd_present(window, film.get_image())
        graphics_core.submit_command_context(context)
        graphics.glfw_poll_events()

    print(scene.vis_scene, scene.sol_scene)

    graphics_core.wait_gpu()


if __name__ == "__main__":
    main()
