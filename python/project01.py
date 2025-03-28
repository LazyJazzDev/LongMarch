import math
import time
from math import isnan
from typing import Any, SupportsFloat

import long_march
from gymnasium.core import ObsType, ActType
from long_march.grassland import graphics
from long_march.snow_mount import solver
from long_march.snow_mount import visualizer
from long_march.grassland.math import rotation
import glfw
import open3d as o3d

import gymnasium as gym

import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices, VecEnvObs, VecEnvStepReturn

from camera_controller import CameraController

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
    def __init__(self, scene: Scene, vertices, indices, bending_stiffness=0.01, elastic_limit=4.0, mesh_mass=1.0,
                 young=3000):
        self.scene = scene
        self.cloth_object = solver.ObjectPack.create_from_mesh(vertices, indices, young=young,
                                                               bending_stiffness=bending_stiffness,
                                                               elastic_limit=elastic_limit, mesh_mass=mesh_mass)
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
        self.open = -0.2

    def pre_solver_update(self, dt):
        gripper_distance_speed = 0.0
        if self.open > 0.0:
            if self.distance < 0.1:
                self.distance += self.open * dt
                gripper_distance_speed = self.open
                self.distance = np.clip(self.distance, 0.01, 0.1)
            else:
                self.distance = 0.1
        else:
            if self.distance > 0.01:
                self.distance += self.open * dt
                gripper_distance_speed = self.open
                self.distance = np.clip(self.distance, 0.01, 0.1)
            else:
                self.distance = 0.01
        self.t += self.R @ self.local_vel * dt
        angular_vel = self.R @ self.local_angular_vel
        self.R = rotation(angular_vel * dt) @ self.R
        min_y_boundary = 0.018
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
        self.left_gripper.set_transform(self.R, self.t + self.R @ np.asarray([-self.distance, 0., 0.]))
        self.right_gripper.set_transform(self.R, self.t + self.R @ np.asarray([self.distance, 0., 0.]))
        self.left_gripper.set_movement(
            self.R @ (local_vel + np.asarray([-gripper_distance_speed, 0., 0.]) + np.cross(self.local_angular_vel,
                                                                                           np.asarray(
                                                                                               [-self.distance, 0.,
                                                                                                0.]))),
            angular_vel)
        self.right_gripper.set_movement(
            self.R @ (local_vel - np.asarray([-gripper_distance_speed, 0., 0.]) + np.cross(self.local_angular_vel,
                                                                                           np.asarray(
                                                                                               [self.distance, 0.,
                                                                                                0.]))),
            angular_vel)

    def set_action(self, local_vel, local_angular_vel, gripper_open):
        gripper_distance_speed_max = 0.2
        self.local_vel = np.asarray(local_vel)
        self.local_angular_vel = local_angular_vel
        self.open = np.clip(gripper_open, -1.0, 1.0) * gripper_distance_speed_max




class Environment:
    def __init__(self, vis_core: visualizer.Core):

        # cloth_vertices = []
        # n_precision = 50
        # for i in range(n_precision):
        #     for j in range(n_precision):
        #         cloth_vertices.append([math.fabs(i / (n_precision - 1) - 0.5), 0.11, j / (n_precision - 1) - 0.5])
        # cloth_vertices = np.asarray(cloth_vertices)
        # np.save("assets/proj01/cloth_target_poses.npy", cloth_vertices)

        self.target_poses = np.load(long_march.grassland.util.find_asset_file("assets/proj01/cloth_target_poses.npy"))
        # print("target_poses: {}".format(self.target_poses))
        self.vis_core = vis_core
        self.scene = None
        self.cloth_initial_vertices = None
        self.cloth_initial_indices = None
        self.gripper = None
        self.ground_object = None
        self.cloth_object = None
        self.ambient_light = self.vis_core.create_entity_ambient_light([0.5, 0.5, 0.5])

        self.directional_light = self.vis_core.create_entity_directional_light([3., 1., 2.], [0.5, 0.5, 0.5])
        self.reset()

    def reset(self):
        self.scene = Scene(self.vis_core)
        self.scene.vis_scene.add_entity(self.ambient_light)
        self.scene.vis_scene.add_entity(self.directional_light)
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

        self.cloth_initial_vertices = cloth_vertices
        self.cloth_initial_indices = cloth_indices

        self.gripper = GripperObject(self.scene)
        self.gripper.t = [-0.35, 0., 0.]
        self.gripper.R = rotation([0., 0., math.pi * -0.5])
        # (self.scene, mesh_vertices * 0.5, mesh_indices)

        self.ground_object = RigidObject(self.scene, mesh_vertices * 10.0, mesh_indices, stiffness=1e6)
        self.cloth_object = GridCloth(self.scene, cloth_vertices, cloth_indices, bending_stiffness=0.003,
                                      elastic_limit=math.pi * 0.1, mesh_mass=0.1, young=300)

        self.scene.build_scene()


        self.ground_object.set_transform(np.identity(3), [0, -10, 0])

    def reward(self):
        poses = np.asarray(self.scene.sol_scene_dev.get_positions(self.cloth_object.cloth_object_view.particle_ids))
        # find max absolute value of poses
        if np.isnan(poses).any():
            return -20
        max_value = np.max(np.abs(poses))
        if max_value > 2.0:
            return -20
        dist = np.linalg.norm(poses - self.target_poses)
        return max(10 - dist, -20) if not isnan(dist) else -20

    def render(self, context, camera, film):
        self.vis_core.render(context, self.scene.vis_scene, camera, film)

    def pre_solver_update(self, dt):
        self.gripper.pre_solver_update(dt)

    def post_solver_update(self, dt, copy_for_vis=True):
        if copy_for_vis:
            self.cloth_object.post_solver_update()


def update_env(env: Environment, dt):
    env.pre_solver_update(dt)
    solver.update_scene(env.scene.sol_scene_dev, dt)
    env.post_solver_update(dt)


def update_envs(envs, dt, render_mode="human"):
    sol_scene_devs = []

    for env in envs:
        env.pre_solver_update(dt)
    for env in envs:
        sol_scene_devs.append(env.scene.sol_scene_dev)
    solver.update_scene_batch(sol_scene_devs, dt)
    for i, env in enumerate(envs):
        env.post_solver_update(dt, copy_for_vis=(render_mode == "human" or (render_mode == "first" and i == 0)))


class PaperEnv(gym.Env):
    def __init__(self, vis_core: visualizer.Core, render_mode="human"):
        self.render_mode = render_mode
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -math.radians(360), -math.radians(360), -math.radians(360), -1.0],
                         dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, math.radians(360), math.radians(360), math.radians(360), 1.0],
                          dtype=np.float32), dtype=np.float32)
        # observation space is a 2500\times 3 matrix, where all the elements has unlimited range
        # another observation space is the gripper position and orientation and the distance between the gripper
        self.observation_space = gym.spaces.Dict(
            {
                "gripper_position": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "gripper_rotation": gym.spaces.Box(low=-1.0, high=1.0, shape=(3, 3), dtype=np.float32),
                "gripper_distance": gym.spaces.Box(low=0.01, high=0.1, shape=(1,), dtype=np.float32),
                "cloth": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2500, 3), dtype=np.float32),
            }
        )

        self.vis_core = vis_core
        self.graphics_core = self.vis_core.get_core()
        if self.render_mode == "human":
            self.camera = self.vis_core.create_camera(
                proj=visualizer.perspective(math.radians(60), 1280 / 720, 0.1, 100.0))
            self.window = self.graphics_core.create_window(1280, 720, "Project01")
            self.film = self.vis_core.create_film(1280, 720)
            self.camera_controller = CameraController(self.window, self.camera)
        self.env = Environment(self.vis_core)
        self.step_count = 0
        self.step_limit = 200

    def get_observation(self):
        poses = np.asarray(
            self.env.scene.sol_scene_dev.get_positions(self.env.cloth_object.cloth_object_view.particle_ids))
        return {
            "gripper_position": self.env.gripper.t,
            "gripper_rotation": self.env.gripper.R,
            "gripper_distance": np.asarray([self.env.gripper.distance]),
            "cloth": poses
        }

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.env.reset()
        obs = self.get_observation()
        self.step_count = 0
        return obs, {}

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.env.gripper.set_action(action[:3], action[3:6], action[6])
        for _ in range(5):
            update_envs([self.env], 0.003, self.render_mode == "human")
            if self.render_mode == "human":
                self.render()
        obs = self.get_observation()
        reward = self.env.reward()
        truncated = False
        self.step_count += 1
        if self.step_count >= self.step_limit:
            truncated = True
        print("reward: {} n_step: {}".format(reward, self.step_count))
        return obs, reward, reward > 5.0, truncated, {}

    def render(self):
        self.camera_controller.update(True)
        context = self.graphics_core.create_command_context()
        self.env.render(context, self.camera, self.film)
        context.cmd_present(self.window, self.film.get_image())
        self.graphics_core.submit_command_context(context)
        graphics.glfw_poll_events()


class PaperVecEnv(VecEnv):
    def __init__(self, vis_core: visualizer.Core, num_envs: int = 2, render_mode="human"):
        action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -math.radians(360), -math.radians(360), -math.radians(360), -1.0],
                         dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, math.radians(360), math.radians(360), math.radians(360), 1.0],
                          dtype=np.float32), dtype=np.float32)
        observation_space = gym.spaces.Dict(
            {
                "gripper_position": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "gripper_rotation": gym.spaces.Box(low=-1.0, high=1.0, shape=(3, 3), dtype=np.float32),
                "gripper_distance": gym.spaces.Box(low=0.01, high=0.1, shape=(1,), dtype=np.float32),
                "cloth": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2500, 3), dtype=np.float32),
            }
        )
        self.vis_core = vis_core
        self.render_modes = [render_mode for i in range(num_envs)]
        super(PaperVecEnv, self).__init__(num_envs=num_envs, action_space=action_space,
                                          observation_space=observation_space)
        # observation space is a 2500\times 3 matrix, where all the elements has unlimited range
        # another observation space is the gripper position and orientation and the distance between the gripper

        self.graphics_core = self.vis_core.get_core()

        self.camera = self.vis_core.create_camera(proj=visualizer.perspective(math.radians(60), 1280 / 720, 0.1, 100.0))
        if render_mode == "human":
            self.windows = [self.graphics_core.create_window(1280, 720, "Project01") for i in range(num_envs)]
            self.camera_controller = CameraController(self.windows[0], self.camera)
            self.camera_controller.update(True)
        elif render_mode == "first":
            self.window = self.graphics_core.create_window(1280, 720, "Project01")
            self.camera_controller = CameraController(self.window, self.camera)
            self.camera_controller.update(True)
        self.films = [self.vis_core.create_film(1280, 720) for i in range(num_envs)]
        self.envs = [PaperEnv(self.vis_core, render_mode="off") for i in range(num_envs)]
        self.n_steps = [0 for i in range(num_envs)]
        self.num_envs = num_envs

    def close(self) -> None:
        print("close")

    def env_is_wrapped(self, wrapper_class: type[gym.Wrapper], indices: VecEnvIndices = None) -> list[bool]:
        return [False for i in range(self.num_envs)]

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> list[Any]:
        print("env_method")
        print("method_name:", method_name)
        print("*method_args:", *method_args)
        print("indices:", indices)
        time.sleep(100)

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        print("get_attr", attr_name, indices)
        if attr_name == "render_mode":
            return self.render_modes

    def get_observation(self):
        original_obs = [env.get_observation() for env in self.envs]
        # merge the dictionaries
        obs = {}
        for key in original_obs[0].keys():
            obs[key] = np.asarray([original_obs[i][key] for i in range(self.num_envs)])
        return obs

    def reset(self) -> VecEnvObs:
        print("reset")
        for env in self.envs:
            env.reset()

        return self.get_observation()

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        print("set_attr")

    def step_async(self, actions: np.ndarray) -> None:
        for i in range(self.num_envs):
            self.envs[i].env.gripper.set_action(actions[i][:3], actions[i][3:6], actions[i][6])
        for substep in range(15):
            envs = [self.envs[i].env for i in range(self.num_envs)]
            update_envs(envs, 0.001, self.render_mode)
            if substep % 5 == 4:
                context = self.graphics_core.create_command_context()
                for i in range(self.num_envs):
                    if self.render_mode == "human":
                        self.envs[i].env.render(context, self.camera, self.films[i])
                        context.cmd_present(self.windows[i], self.films[i].get_image())
                if self.render_mode == "first":
                    self.envs[0].env.render(context, self.camera, self.films[0])
                    context.cmd_present(self.window, self.films[0].get_image())
                context.submit()
                graphics.glfw_poll_events()

        context = self.graphics_core.create_command_context()
        for i in range(self.num_envs):
            if self.render_mode == "human":
                self.envs[i].env.render(context, self.camera, self.films[i])
                context.cmd_present(self.windows[i], self.films[i].get_image())
        if self.render_mode == "first":
            self.envs[0].env.render(context, self.camera, self.films[0])
            context.cmd_present(self.window, self.films[0].get_image())
        context.submit()
        graphics.glfw_poll_events()
        for i in range(self.num_envs):
            self.envs[i].step_count += 1

    def step_wait(self) -> VecEnvStepReturn:
        rewards = np.asarray([env.env.reward() for env in self.envs])
        dones = [False for reward in rewards]
        truncated = [env.step_count >= env.step_limit for env in self.envs]
        # merge dones and truncated
        merged_dones = [dones[i] or truncated[i] for i in range(self.num_envs)]
        # make an empty tuple name infos
        infos = []

        for i in range(self.num_envs):
            info = {"TimeLimit.truncated": truncated[i]}
            ob = self.envs[i].get_observation()
            if rewards[i] <= -19.:
                merged_dones[i] = True
                dones[i] = True
                rewards[i] = -(self.envs[i].step_limit - self.envs[i].step_count) * 30

            if merged_dones[i]:
                # if dones[i]:
                #     rewards[i] += (self.envs[i].step_limit - self.envs[i].step_count) * 10
                info["terminal_observation"] = ob
                self.envs[i].reset()
            infos.append(info)
        obs = self.get_observation()
        return obs, np.asarray(rewards, dtype=np.float32), np.asarray(merged_dones), infos


def main():
    core_settings = graphics.CoreSettings()
    core_settings.frames_in_flight = 1

    graphics_core = graphics.Core(graphics.BACKEND_API_D3D12, core_settings)
    vis_core = visualizer.Core(graphics_core)

    vec_env = PaperVecEnv(vis_core, 64,
                          render_mode="first")  # SubprocVecEnv([make_env_custom() for i in range(num_cpu)])
    model = PPO("MultiInputPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=10_000_000, progress_bar=True)
    for i in range(10000):
        model.save("tmp/model_{}.zip".format(i))

    obs = vec_env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render()

    window = graphics_core.create_window(1280, 720, "Project01")

    camera = vis_core.create_camera(proj=visualizer.perspective(math.radians(60), 1280 / 720, 0.1, 100.0))
    camera_controller = CameraController(window, camera)

    film = vis_core.create_film(1280, 720)

    envs = [Environment(vis_core) for i in range(1)]

    control_camera = False

    def key_callback(key, scancode, action, mods):
        nonlocal control_camera
        if key == glfw.KEY_TAB and action == glfw.PRESS:
            control_camera = not control_camera

    window.reg_key_callback(key_callback)

    # for env in envs:
    #     env.gripper.t = np.asarray([-1.5, 0., 0.])

    n_step = 0
    camera_controller.update(True)
    while not window.should_close():
        camera_controller.update(control_camera)
        if not control_camera:
            k_angular_speed = math.radians(360)
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
            gripper_open = 1.0
            if window.get_key(glfw.KEY_SPACE) == glfw.PRESS:
                gripper_open = -1.0
            for env in envs:
                env.gripper.set_action(gripper_vel, gripper_angular_vel, gripper_open)

        update_envs(envs, 0.003)
        context = graphics_core.create_command_context()
        envs[0].render(context, camera, film)
        context.cmd_present(window, film.get_image())
        graphics_core.submit_command_context(context)
        graphics.glfw_poll_events()
        n_step += 1
        print("reward: {} n_step: {}".format(envs[0].reward(), n_step))

    graphics_core.wait_gpu()
    # poses = np.asarray(envs[0].scene.sol_scene_dev.get_positions(envs[0].cloth_object.cloth_object_view.particle_ids))
    # print(poses, poses.shape, type(poses))
    # np.save("assets/proj01/cloth_target_poses.npy", poses)


if __name__ == "__main__":
    main()
