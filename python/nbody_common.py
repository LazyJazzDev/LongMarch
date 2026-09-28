"""Shared pieces of the Python nbody_cs and nbody_warp demos.

Both mirror demo/nbody_cs and demo/nbody_cuda: the same parameters (params.h), initial
galaxies, particle and HDR shaders, ImGui panel and mouse rotation. Only the simulation
differs: a Slang compute shader or a Warp kernel.
"""

import collections
import math
import pathlib
import time

import numpy as np
from long_march import graphics
from long_march.graphics import imgui

from glm_math import as_bytes, look_at, perspective, rotate

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# params.h
NUM_PARTICLE = 65536
DELTA_T = 3e-2
GRAVITY_COE = 1e2 / NUM_PARTICLE
INITIAL_SPEED = 2.0
INITIAL_RADIUS = 10.0
PARTICLE_SIZE = 0.07

GLFW_MOUSE_BUTTON_LEFT = 0
# From NVIDIA's official CUDA N-body example.
NUM_FLOPS_PER_INTERACTION = 20.0


def log_info(message):
    print(f"[info] {message}", flush=True)


def log_warning(message):
    print(f"[warning] {message}", flush=True)


class FPSCounter:
    """grassland::FPSCounter: frames over a sliding window of about one second."""

    def __init__(self):
        self.frames = collections.deque()

    def tick_frame(self):
        now = time.perf_counter()
        self.frames.append(now)
        while len(self.frames) > 2 and self.frames[0] < now - 1.0:
            self.frames.popleft()

    def get_fps(self):
        if len(self.frames) < 2:
            return 0.0
        return (len(self.frames) - 1) / (self.frames[-1] - self.frames[0])

    def tick_fps(self):
        self.tick_frame()
        return self.get_fps()


def random_in_sphere(rng, count):
    z = rng.random(count, np.float32) * 2.0 - 1.0
    inv_z = np.sqrt(1.0 - z * z)
    theta = rng.random(count, np.float32) * np.float32(math.pi * 2.0)
    on_sphere = np.stack([inv_z * np.sin(theta), inv_z * np.cos(theta), z], axis=1)
    return on_sphere * np.power(rng.random(count, np.float32), np.float32(0.333333333333333333))[:, None]


def initial_particles(rng, n_particles, galaxy_number):
    """Particles in galaxy_number clusters, with zero mean cluster position and velocity."""
    origins = random_in_sphere(rng, galaxy_number) * (INITIAL_RADIUS * 2.0)
    initial_vels = random_in_sphere(rng, galaxy_number) * (INITIAL_RADIUS * 0.1)
    origins -= origins.mean(axis=0)
    initial_vels -= initial_vels.mean(axis=0)
    index = rng.integers(0, galaxy_number, n_particles)
    radius = INITIAL_RADIUS * 0.2 * (10.0 / galaxy_number) ** (1.0 / 3.0)
    positions = random_in_sphere(rng, n_particles) * np.float32(radius) + origins[index]
    velocities = random_in_sphere(rng, n_particles) * np.float32(INITIAL_SPEED) + initial_vels[index]
    return positions.astype(np.float32), velocities.astype(np.float32)


def speed_histogram(velocities, num_samples=100):
    speeds = np.linalg.norm(velocities, axis=1)
    max_speed = speeds.max()
    bins = np.clip((speeds / max_speed * num_samples).astype(np.int64), 0, num_samples - 1)
    samples = np.bincount(bins, minlength=num_samples)
    return (samples / samples.max()).astype(np.float32).tolist()


def format_flops(ops):
    if ops < 8e2:
        return f"{ops:.2f} FLOP/s"
    if ops < 8e5:
        return f"{ops * 1e-3:.2f} KFLOP/s"
    if ops < 8e8:
        return f"{ops * 1e-6:.2f} MFLOP/s"
    return f"{ops * 1e-9:.2f} GFLOP/s"


class ParticleRenderer:
    """Additive particle sprites followed by the HDR/sRGB resolve pass, as in the C++ demos."""

    def __init__(self, core, shader_directory, width, height):
        self.core = core
        self.uniform_buffer = core.create_buffer(136, graphics.BUFFER_TYPE_DYNAMIC)  # GlobalUniformObject
        self.frame_image = core.create_image(width, height, graphics.IMAGE_FORMAT_R32G32B32A32_SFLOAT)
        directory = str(shader_directory)
        self.vertex_shader = core.create_shader_from_directory(directory, "particle.slang", "VSMain", "vs_6_0")
        self.fragment_shader = core.create_shader_from_directory(directory, "particle.slang", "PSMain", "ps_6_0")
        self.program = core.create_program([self.frame_image.format()], graphics.IMAGE_FORMAT_UNDEFINED)
        self.program.set_blend_state(0, graphics.BlendState(
            graphics.BLEND_FACTOR_ONE, graphics.BLEND_FACTOR_ONE, graphics.BLEND_OP_ADD,
            graphics.BLEND_FACTOR_ONE, graphics.BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, graphics.BLEND_OP_ADD))
        self.program.add_input_binding(12, True)
        self.program.add_input_attribute(0, graphics.INPUT_TYPE_FLOAT3, 0)
        self.program.add_resource_binding(graphics.RESOURCE_TYPE_UNIFORM_BUFFER, 1)
        self.program.bind_shader(self.vertex_shader, graphics.SHADER_TYPE_VERTEX)
        self.program.bind_shader(self.fragment_shader, graphics.SHADER_TYPE_PIXEL)
        self.program.finalize()

        self.hdr_vertex_shader = core.create_shader_from_directory(directory, "hdr.slang", "VSMain", "vs_6_0")
        self.hdr_fragment_shader = core.create_shader_from_directory(directory, "hdr.slang", "PSMain", "ps_6_0")
        self.hdr_program = core.create_program([], graphics.IMAGE_FORMAT_UNDEFINED)
        self.hdr_program.add_resource_binding(graphics.RESOURCE_TYPE_UNIFORM_BUFFER, 1)
        self.hdr_program.add_resource_binding(graphics.RESOURCE_TYPE_WRITABLE_IMAGE, 1)
        self.hdr_program.bind_shader(self.hdr_vertex_shader, graphics.SHADER_TYPE_VERTEX)
        self.hdr_program.bind_shader(self.hdr_fragment_shader, graphics.SHADER_TYPE_PIXEL)
        self.hdr_program.finalize()

    def resize(self, width, height):
        self.core.wait_gpu()
        if width <= 0 or height <= 0:
            return
        self.frame_image = None
        self.frame_image = self.core.create_image(width, height, graphics.IMAGE_FORMAT_R32G32B32A32_SFLOAT)

    def update_uniforms(self, rotation, hdr, extent=None):
        width, height = extent or (self.frame_image.extent().width, self.frame_image.extent().height)
        world_to_cam = look_at((10.0, 20.0, 30.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)) @ rotation
        world_to_screen = perspective(math.radians(60.0), width / height, 0.1, 100.0) @ world_to_cam
        tail = np.array([PARTICLE_SIZE], np.float32).tobytes() + np.array([int(hdr)], np.int32).tobytes()
        self.uniform_buffer.upload_data(as_bytes(world_to_screen, np.linalg.inv(world_to_cam)) + tail)

    def record(self, ctx, positions, n_particles):
        extent = self.frame_image.extent()
        ctx.cmd_clear_image(self.frame_image, [0.0, 0.0, 0.0, 0.0])
        ctx.cmd_begin_rendering([self.frame_image], None)
        ctx.cmd_bind_program(self.program)
        ctx.cmd_set_primitive_topology(graphics.PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        ctx.cmd_set_scissor(0, 0, extent.width, extent.height)
        ctx.cmd_set_viewport(0, 0, extent.width, extent.height)
        ctx.cmd_bind_vertex_buffers(0, [positions], [0])
        ctx.cmd_bind_resources(0, [self.uniform_buffer])
        ctx.cmd_draw(6, n_particles, 0, 0)
        ctx.cmd_end_rendering()
        ctx.cmd_begin_rendering([], None)
        ctx.cmd_bind_program(self.hdr_program)
        ctx.cmd_bind_resources(0, [self.uniform_buffer])
        ctx.cmd_bind_resources(1, [self.frame_image])
        ctx.cmd_draw(6, 1, 0, 0)
        ctx.cmd_end_rendering()


class NBodyControls:
    """Mouse rotation and the ImGui panel shared by both demos."""

    def __init__(self, window, title, backend=None):
        self.window = window
        self.title = title
        self.backend = backend
        self.rotation = np.identity(4, np.float32)
        self.last_cursor = None
        self.hdr = False
        self.step = True
        self.delta_t = DELTA_T
        self.galaxy_number = 10
        self.last_frame_tp = None
        window.register_mouse_move_event(self.on_mouse_move)

    def on_mouse_move(self, xpos, ypos):
        imgui.set_current_context(self.window)
        if self.last_cursor is None:
            self.last_cursor = (xpos, ypos)
        if self.window.is_mouse_button_down(GLFW_MOUSE_BUTTON_LEFT) and not imgui.want_capture_mouse():
            diffx, diffy = xpos - self.last_cursor[0], ypos - self.last_cursor[1]
            self.rotation = rotate(math.radians(diffx), (0.0, 1.0, 0.0)) @ self.rotation
            self.rotation = rotate(math.radians(diffy), (1.0, 0.0, 0.0)) @ self.rotation
        self.last_cursor = (xpos, ypos)

    def update_imgui(self, n_particles, download_velocities, reset):
        """Draw the panel; download_velocities() and reset() are called on demand."""
        self.window.begin_imgui_frame()
        imgui.set_next_window_pos(0.0, 0.0, imgui.COND_ONCE)
        imgui.set_next_window_bg_alpha(0.3)
        trigger_hdr_switch = False
        if imgui.begin(self.title, imgui.WINDOW_FLAGS_NO_MOVE):
            imgui.text("Statistics")
            imgui.separator()
            if self.backend:
                imgui.text(f"Backend: {self.backend}")
            current_tp = time.perf_counter()
            if self.last_frame_tp is None:
                self.last_frame_tp = current_tp
            duration_ms = (current_tp - self.last_frame_tp) * 1e3
            imgui.text(f"Frame Duration: {duration_ms:.3f} ms")
            if self.step:
                interactions_per_second = n_particles * n_particles / max(duration_ms * 1e-3, 1e-9)
                imgui.text(format_flops(interactions_per_second * NUM_FLOPS_PER_INTERACTION))

            if imgui.collapsing_header("Speed Distribution"):
                imgui.plot_lines("##1", speed_histogram(download_velocities()), 0.0, 1.0)

            imgui.new_line()

            imgui.text("Control")
            imgui.separator()
            if imgui.button("Reset"):
                reset()
            imgui.same_line()
            if imgui.button("Pause" if self.step else "Resume"):
                self.step = not self.step
            _, self.delta_t = imgui.slider_float("Delta Time", self.delta_t, 0.001, 0.1, "%.3f",
                                                 imgui.SLIDER_FLAGS_LOGARITHMIC)
            _, self.galaxy_number = imgui.slider_int("Galaxy Number", self.galaxy_number, 1, 20, "%d")
            imgui.new_line()

            imgui.text("Visualizer")
            imgui.separator()
            if imgui.button("HDR: " + ("ON" if self.hdr else "OFF")):
                trigger_hdr_switch = True
            self.last_frame_tp = current_tp
        imgui.end()
        self.window.end_imgui_frame()
        if trigger_hdr_switch:
            if self.window.set_hdr(not self.hdr) == 0:
                self.hdr = not self.hdr
            else:
                log_warning("HDR mode change unavailable; keeping the current presentation mode.")
