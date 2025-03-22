import math
import time
import long_march
from long_march.grassland import graphics
from long_march.snow_mount import solver
from long_march.snow_mount import visualizer
from long_march.grassland.math import rotation
import glfw
import open3d as o3d

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
    def __init__(self, scene: Scene, vertices, indices, stiffness=1e5):
        self.scene = scene
        self.vertices = np.asarray(vertices)
        self.indices = np.asarray(indices)
        self.vis_mesh = scene.vis_scene.get_core().create_mesh()
        self.vis_mesh.set_vertices(vertices)
        self.vis_mesh.set_indices(indices)
        mesh_sdf = long_march.grassland.math.MeshSDF(vertices, indices)
        self.sol_rigid_object = solver.RigidObject(mesh_sdf, stiffness=stiffness)
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

    def set_movement(self, v, omega):
        rigid_object_state = self.scene.sol_scene_dev.get_rigid_object_state(self.sol_rigid_id)
        rigid_object_state.v = v
        rigid_object_state.omega = omega
        self.scene.sol_scene_dev.set_rigid_object_state(self.sol_rigid_id, rigid_object_state)

    def set_color(self, color):
        self.vis_entity.set_material(visualizer.Material(color))

    @staticmethod
    def load_from_mesh(scene: Scene, path: str, stiffness=1e5):
        full_path = long_march.grassland.util.find_asset_file(path)
        # load obj mesh from full_path, use open3d
        mesh = o3d.io.read_triangle_mesh(full_path)
        vertices = np.asarray(mesh.vertices)
        indices = np.asarray(mesh.triangles).flatten()
        return RigidObject(scene, vertices, indices, stiffness=stiffness)


class GridCloth:
    def __init__(self, scene: Scene, vertices, indices):
        self.scene = scene
        self.cloth_object = solver.ObjectPack.create_from_mesh(vertices, indices, bending_stiffness=0.003,
                                                               elastic_limit=math.pi * 0.05)
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


class GripperObject:
    def __init__(self, scene: Scene):
        self.scene = scene
        self.left_gripper = RigidObject.load_from_mesh(scene, "meshes/gripper_left.obj", stiffness=1e6)
        self.right_gripper = RigidObject.load_from_mesh(scene, "meshes/gripper_right.obj", stiffness=1e6)
        self.left_gripper.set_color([0.5, 0.5, 0.5, 1.0])
        self.right_gripper.set_color([0.5, 0.5, 0.5, 1.0])
        self.R = np.identity(3)
        self.t = np.asarray([0., 0., 0.])
        self.local_angular_vel = np.asarray([0., 0., 0.])
        self.local_vel = np.asarray([0., 0., 0.])
        self.distance = 0.01
        self.open = False

    def pre_solver_update(self, dt):
        gripper_distance_speed_max = 0.2
        gripper_distance_speed = 0.0
        print("open: ", self.open)
        if self.open:
            if self.distance < 0.1:
                self.distance += gripper_distance_speed_max * dt
                gripper_distance_speed = gripper_distance_speed_max
                self.distance = np.clip(self.distance, 0.01, 0.1)
            else:
                self.distance = 0.1
        else:
            if self.distance > 0.01:
                self.distance -= gripper_distance_speed_max * dt
                gripper_distance_speed = -gripper_distance_speed_max
                self.distance = np.clip(self.distance, 0.01, 0.1)
            else:
                self.distance = 0.01
        self.t += self.R @ self.local_vel * dt
        angular_vel = self.R @ self.local_angular_vel
        self.R = rotation(angular_vel * dt) @ self.R
        self.left_gripper.set_transform(self.R, self.t + self.R @ np.asarray([-self.distance, 0., 0.]))
        self.right_gripper.set_transform(self.R, self.t + self.R @ np.asarray([self.distance, 0., 0.]))
        min_y_boundary = 0.015
        coord_y_correction = 0
        # enumerate all the transformed vertices of the gripper
        for vertex in self.left_gripper.vertices:
            v = self.R @ (vertex + np.asarray([-self.distance, 0., 0.])) + self.t
            coord_y_correction = max(coord_y_correction, min_y_boundary - v[1])
        for vertex in self.right_gripper.vertices:
            v = self.R @ (vertex + np.asarray([self.distance, 0., 0.])) + self.t
            coord_y_correction = max(coord_y_correction, min_y_boundary - v[1])
        self.t[1] += coord_y_correction
        local_vel = self.local_vel + np.transpose(self.R) @ np.asarray([0., coord_y_correction / dt, 0.])
        self.left_gripper.set_movement(
            self.R @ (local_vel + np.asarray([-gripper_distance_speed, 0., 0.]) + np.cross(self.local_angular_vel, np.asarray([-self.distance, 0., 0.]))),
            angular_vel)
        self.right_gripper.set_movement(
            self.R @ (local_vel - np.asarray([-gripper_distance_speed, 0., 0.]) + np.cross(self.local_angular_vel, np.asarray([self.distance, 0., 0.]))),
            angular_vel)


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
        if effective:
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

            R[:3, :3] = rotation([0, self.camera_orientation[1], 0]) @ rotation(
                [self.camera_orientation[0], 0, 0]) @ rotation([0, 0, self.camera_orientation[2]])

            self.camera_position += R[:3, :3] @ move_signal

            R[:3, 3] = self.camera_position

            # inverse matrix R
            R = np.linalg.inv(R)

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
        for i in range(50):
            for j in range(50):
                cloth_vertices.append([i / 49 - 0.5, 2.0, j / 49 - 0.5])
        cloth_vertices = np.asarray(cloth_vertices)

        cloth_indices = []
        for i in range(49):
            i1 = i + 1
            for j in range(49):
                j1 = j + 1
                v00 = i * 50 + j
                v01 = i * 50 + j1
                v10 = i1 * 50 + j
                v11 = i1 * 50 + j1
                cloth_indices.append([v00, v01, v11])
                cloth_indices.append([v00, v11, v10])
        # serialize the indices
        cloth_indices = np.asarray(cloth_indices).flatten()

        self.gripper = GripperObject(self.scene)
        # (self.scene, mesh_vertices * 0.5, mesh_indices)

        self.ground_object = RigidObject(self.scene, mesh_vertices * 10.0, mesh_indices)
        self.cloth_object = GridCloth(self.scene, cloth_vertices, cloth_indices)

        self.scene.build_scene()

        self.ground_object.set_transform(np.identity(3), [0, -10, 0])

    def render(self, context, camera, film):
        self.vis_core.render(context, self.scene.vis_scene, camera, film)

    def pre_solver_update(self, dt):
        self.gripper.pre_solver_update(dt)

    def post_solver_update(self, dt):
        self.cloth_object.post_solver_update()


def update_env(env: Environment, dt):
    env.pre_solver_update(dt)
    solver.update_scene(env.scene.sol_scene_dev, dt)
    env.post_solver_update(dt)


def update_envs(envs, dt):
    sol_scene_devs = []

    for env in envs:
        env.pre_solver_update(dt)
    for env in envs:
        sol_scene_devs.append(env.scene.sol_scene_dev)
    solver.update_scene_batch(sol_scene_devs, dt)
    for env in envs:
        env.post_solver_update(dt)


def main():
    core_settings = graphics.CoreSettings()
    core_settings.frames_in_flight = 1

    graphics_core = graphics.Core(graphics.BACKEND_API_VULKAN, core_settings)
    vis_core = visualizer.Core(graphics_core)

    window = graphics_core.create_window(1280, 720, "Project01")

    camera = vis_core.create_camera(proj=visualizer.perspective(math.radians(60), 1280 / 720, 0.1, 100.0))
    camera_controller = CameraController(window, camera)

    film = vis_core.create_film(1280, 720)

    envs = [Environment(vis_core) for i in range(1)]

    control_camera = True

    def key_callback(key, scancode, action, mods):
        nonlocal control_camera
        if key == glfw.KEY_TAB and action == glfw.PRESS:
            control_camera = not control_camera

    window.reg_key_callback(key_callback)

    for env in envs:
        env.gripper.t = np.asarray([-1.5, 0., 0.])

    while not window.should_close():
        camera_controller.update(control_camera)
        if not control_camera:
            k_angular_speed = math.radians(180)
            k_speed = 1.0
            gripper_vel = np.asarray([0., 0., 0.])
            gripper_angular_vel = np.asarray([0., 0., 0.])
            if window.get_key(glfw.KEY_A) == glfw.PRESS:
                gripper_vel[0] += -k_speed
            if window.get_key(glfw.KEY_D) == glfw.PRESS:
                gripper_vel[0] += k_speed
            if window.get_key(glfw.KEY_W) == glfw.PRESS:
                gripper_vel[2] += -k_speed
            if window.get_key(glfw.KEY_S) == glfw.PRESS:
                gripper_vel[2] += k_speed
            if window.get_key(glfw.KEY_R) == glfw.PRESS:
                gripper_vel[1] += k_speed
            if window.get_key(glfw.KEY_F) == glfw.PRESS:
                gripper_vel[1] -= k_speed
            if window.get_key(glfw.KEY_Q) == glfw.PRESS:
                gripper_angular_vel[2] += k_angular_speed
            if window.get_key(glfw.KEY_E) == glfw.PRESS:
                gripper_angular_vel[2] -= k_angular_speed
            if window.get_key(glfw.KEY_UP) == glfw.PRESS:
                gripper_angular_vel[0] += k_angular_speed
            if window.get_key(glfw.KEY_DOWN) == glfw.PRESS:
                gripper_angular_vel[0] -= k_angular_speed
            if window.get_key(glfw.KEY_LEFT) == glfw.PRESS:
                gripper_angular_vel[1] += k_angular_speed
            if window.get_key(glfw.KEY_RIGHT) == glfw.PRESS:
                gripper_angular_vel[1] -= k_angular_speed
            gripper_open = True
            if window.get_key(glfw.KEY_SPACE) == glfw.PRESS:
                gripper_open = False
            print("gripper_open: ", gripper_open)
            for env in envs:
                env.gripper.local_vel = gripper_vel
                env.gripper.local_angular_vel = gripper_angular_vel
                env.gripper.open = gripper_open
        update_envs(envs, 0.003)
        context = graphics_core.create_command_context()
        envs[0].render(context, camera, film)
        context.cmd_present(window, film.get_image())
        graphics_core.submit_command_context(context)
        graphics.glfw_poll_events()

    graphics_core.wait_gpu()


if __name__ == "__main__":
    main()
