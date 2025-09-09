import math
import time
import scipy
import long_march
from long_march.grassland import graphics
from long_march.snowberg import solver
from long_march.snowberg import visualizer
import glfw

glfw.init()

import numpy as np


class Scene:
    def __init__(self, vis_core: visualizer.Core):
        self.sol_scene_dev = None
        self.vis_scene = vis_core.create_scene()
        self.sol_scene = solver.Scene()

    def build_scene(self):
        self.sol_scene_dev = solver.SceneDevice(self.sol_scene)


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
        rigid_object_state = self.scene.sol_scene_dev.get_rigid_object_state(self.sol_rigid_id)
        rigid_object_state.R = R
        rigid_object_state.t = t
        self.scene.sol_scene_dev.set_rigid_object_state(self.sol_rigid_id, rigid_object_state)


class GridCloth:
    def __init__(self, scene: Scene, vertices, indices):
        self.scene = scene
        self.cloth_object = solver.ObjectPack.create_from_mesh(vertices, indices)
        self.cloth_object_view = scene.sol_scene.add_object(self.cloth_object)
        self.cloth_vis_mesh = scene.vis_scene.get_core().create_mesh()
        self.cloth_vis_mesh.set_vertices(scene.sol_scene.get_positions(self.cloth_object_view.particle_ids))
        self.cloth_vis_mesh.set_indices(indices)
        self.cloth_vis_entity = scene.vis_scene.get_core().create_entity_mesh_object()
        self.cloth_vis_entity.set_mesh(self.cloth_vis_mesh)
        self.cloth_vis_entity.set_transform(np.identity(4))
        self.cloth_vis_entity.set_material(visualizer.Material([0.8, 0.5, 0.5, 1.0]))
        self.scene.vis_scene.add_entity(self.cloth_vis_entity)

    def post_solver_update(self):
        self.cloth_vis_mesh.set_vertices(self.scene.sol_scene_dev.get_positions(self.cloth_object_view.particle_ids))


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


class Environment:
    def __init__(self, vis_core: visualizer.Core):
        self.vis_core = vis_core
        self.scene = Scene(vis_core)
        mesh_vertices = np.asarray([
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1]])
        mesh_indices = np.asarray([0, 2, 1,
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
                                   0, 1, 5])

        cloth_vertices = []
        n_precision = 50
        for i in range(n_precision):
            for j in range(n_precision):
                cloth_vertices.append([i / (n_precision - 1) - 0.5, 0.12, j / (n_precision - 1) - 0.5])
        cloth_vertices = np.asarray(cloth_vertices)

        cloth_indices = []
        for i in range(n_precision - 1):
            i1 = i + 1
            for j in range(n_precision - 1):
                j1 = j + 1
                v00 = i * n_precision + j
                v01 = i * n_precision + j1
                v10 = i1 * n_precision + j
                v11 = i1 * n_precision + j1
                cloth_indices.append([v00, v01, v11])
                cloth_indices.append([v00, v11, v10])
        # serialize the indices
        cloth_indices = np.asarray(cloth_indices).flatten()

        self.rigid_object = RigidObject(self.scene, mesh_vertices * 0.5, mesh_indices)

        self.ground_object = RigidObject(self.scene, mesh_vertices * 10.0, mesh_indices)
        self.cloth_object = GridCloth(self.scene, cloth_vertices, cloth_indices)

        self.scene.build_scene()

        self.rigid_object.set_transform(np.identity(3), [-0.5, -0.5, 0])
        self.ground_object.set_transform(np.identity(3), [0, -11, 0])
    def render(self, context, camera, film):
        self.vis_core.render(context, self.scene.vis_scene, camera, film)

def update_env(env: Environment, dt):
    solver.update_scene(env.scene.sol_scene_dev, dt)
    env.cloth_object.post_solver_update()

def update_envs(envs, dt):
    sol_scene_devs = []
    for env in envs:
        sol_scene_devs.append(env.scene.sol_scene_dev)
    solver.update_scene_batch(sol_scene_devs, dt)
    for env in envs:
        env.cloth_object.post_solver_update()

def main():
    time.sleep(1)
    core_settings = graphics.CoreSettings()
    core_settings.frames_in_flight = 1

    graphics_core = graphics.Core(graphics.BACKEND_API_VULKAN, core_settings)
    vis_core = visualizer.Core(graphics_core)

    envs = [Environment(vis_core) for i in range(256)]

    begin = time.time()
    for i in range(1):
        update_envs(envs, 0.003)
    end = time.time()
    print(f"Time: {end - begin}")

if __name__ == "__main__":
    main()
