"""Python counterpart of demo/nbody_cs: N-body simulation in a Slang compute shader.

Uses the C++ demo's shaders and command line. Benchmark modes (compute, offscreen, window)
wait for the GPU every frame. Use Nsight for profiling.
"""

import sys

import numpy as np
from long_march import graphics
from long_march.graphics import imgui

import nbody_common as nb

SHADER_DIRECTORY = nb.REPO_ROOT / "demo" / "nbody_cs" / "shaders"
MODES = ("interactive", "compute", "offscreen", "window")

HELP = """nbody_cs [--backend auto|metal|vulkan] [--mode interactive|compute|offscreen|window]
  [--particles N] [--frames N] [--warmup N] [--seed N] [--width W] [--height H]
  [--state-output path] [--debug]
Benchmark modes wait for GPU completion each frame; use Nsight for profiling."""


class Options:
    def __init__(self):
        self.backend = graphics.BACKEND_API_DEFAULT
        self.particles = nb.NUM_PARTICLE
        self.frames = 120
        self.warmup = 20
        self.width = 1920
        self.height = 1080
        self.seed = 1
        self.debug = False
        self.mode = "interactive"
        self.state_output = ""


def parse_options(argv):
    options = Options()
    i = 1
    while i < len(argv):
        arg = argv[i]

        def value():
            nonlocal i
            i += 1
            if i >= len(argv):
                raise ValueError("missing value for " + arg)
            return argv[i]

        if arg == "--backend":
            name = value()
            if name == "metal":
                options.backend = graphics.BACKEND_API_METAL
            elif name == "vulkan":
                options.backend = graphics.BACKEND_API_VULKAN
            elif name != "auto":
                raise ValueError("unknown backend: " + name)
        elif arg in ("--particles", "--frames", "--warmup", "--seed", "--width", "--height"):
            setattr(options, arg[2:], int(value()))
        elif arg == "--state-output":
            options.state_output = value()
        elif arg == "--debug":
            options.debug = True
        elif arg == "--mode":
            options.mode = value()
        elif arg == "--help":
            print(HELP)
            return None
        else:
            raise ValueError("unknown argument: " + arg)
        i += 1
    if not graphics.support_backend_api(options.backend):
        raise RuntimeError("backend was not built")
    if options.particles <= 0 or options.particles % 128:
        raise ValueError("particles must be a positive multiple of 128")
    if options.frames <= 0 or options.warmup < 0 or options.width <= 0 or options.height <= 0:
        raise ValueError("invalid frame count or image dimensions")
    if options.mode not in MODES:
        raise ValueError("unknown mode: " + options.mode)
    return options


class NBodyCS:
    def __init__(self, options):
        self.options = options
        self.n_particles = options.particles
        self.rng = np.random.default_rng(options.seed)
        self.core = graphics.Core(options.backend, graphics.CoreSettings(2, options.debug))
        if self.core.init_auto(False):
            raise RuntimeError("could not initialize graphics backend")
        self.window = None
        if options.mode in ("interactive", "window"):
            self.window = self.core.create_window(options.width, options.height, "NBodyCS", False, True)
        self.fps_counter = nb.FPSCounter()

    def benchmark(self):
        return self.options.mode != "interactive"

    def run(self):
        self.on_init()
        print(f"Backend: {graphics.backend_api_string(self.core.api())}, device: {self.core.device_name()}",
              flush=True)
        frame = -self.options.warmup
        while (not self.window or not self.window.should_close()) and (
                not self.benchmark() or frame < self.options.frames):
            self.on_update()
            self.on_render()
            if self.window:
                graphics.Window.poll_events()
            frame += 1
        self.core.wait_gpu()
        if self.options.state_output:
            with open(self.options.state_output, "wb") as output:
                output.write(self.particles_pos.download_data())
                output.write(self.particles_vel.download_data())
        if self.window:
            self.window.terminate_imgui()
        self.on_close()

    def on_init(self):
        self.renderer = None
        if self.options.mode != "compute":
            width, height = (self.window.get_framebuffer_size() if self.window
                             else (self.options.width, self.options.height))
            self.renderer = nb.ParticleRenderer(self.core, SHADER_DIRECTORY, width, height)
        if self.window:
            self.window.register_framebuffer_resize_event(self.renderer.resize)

        size = 12 * self.n_particles
        self.particles_pos = self.core.create_buffer(size, graphics.BUFFER_TYPE_STATIC)
        self.particles_vel = self.core.create_buffer(size, graphics.BUFFER_TYPE_STATIC)
        self.particles_pos_new = self.core.create_buffer(size, graphics.BUFFER_TYPE_STATIC)
        self.global_settings_buffer = self.core.create_buffer(12, graphics.BUFFER_TYPE_DYNAMIC)
        self.controls = None
        if self.window:
            self.window.init_imgui(str(nb.REPO_ROOT / "assets" / "fonts" / "simhei.ttf"), 20.0)
            if self.benchmark():
                imgui.set_current_context(self.window)
                imgui.set_ini_filename(None)
            self.controls = nb.NBodyControls(self.window, "NBodyCS", graphics.backend_api_string(self.core.api()))
        self.reset_particles()

        self.nbody_compute_shader = self.core.create_shader_from_directory(str(SHADER_DIRECTORY), "nbody.slang",
                                                                           "CSMain", "cs_6_0")
        self.nbody_compute_program = self.core.create_compute_program(self.nbody_compute_shader)
        self.nbody_compute_program.add_resource_binding(graphics.RESOURCE_TYPE_STORAGE_BUFFER, 1)
        self.nbody_compute_program.add_resource_binding(graphics.RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER, 1)
        self.nbody_compute_program.add_resource_binding(graphics.RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER, 1)
        self.nbody_compute_program.add_resource_binding(graphics.RESOURCE_TYPE_UNIFORM_BUFFER, 1)
        self.nbody_compute_program.finalize()

    def on_close(self):
        del self.nbody_compute_program, self.nbody_compute_shader, self.renderer
        del self.particles_pos, self.particles_vel, self.particles_pos_new, self.global_settings_buffer

    def galaxy_number(self):
        return self.controls.galaxy_number if self.controls else 10

    def reset_particles(self):
        positions, velocities = nb.initial_particles(self.rng, self.n_particles, self.galaxy_number())
        self.particles_pos.upload_data(positions.tobytes())
        self.particles_vel.upload_data(velocities.tobytes())

    def download_velocities(self):
        return np.frombuffer(self.particles_vel.download_data(), np.float32).reshape(-1, 3)

    def on_update(self):
        if self.controls:
            self.controls.update_imgui(self.n_particles, self.download_velocities, self.reset_particles)
        rotation = self.controls.rotation if self.controls else np.identity(4, np.float32)
        hdr = self.controls.hdr if self.controls else False
        delta_t = self.controls.delta_t if self.controls else nb.DELTA_T
        if self.renderer:
            self.renderer.update_uniforms(rotation, hdr)
        self.global_settings_buffer.upload_data(np.array([self.n_particles], np.int32).tobytes() +
                                                np.array([delta_t, nb.GRAVITY_COE], np.float32).tobytes())
        if self.window and not self.benchmark():
            self.window.set_title(f"NBody Compute Shader FPS: {self.fps_counter.tick_fps():.6f}")

    def on_render(self):
        ctx = self.core.create_command_context()
        if not self.controls or self.controls.step:
            ctx.cmd_bind_compute_program(self.nbody_compute_program)
            ctx.cmd_bind_resources(0, [self.particles_pos], graphics.BIND_POINT_COMPUTE)
            ctx.cmd_bind_resources(1, [self.particles_vel], graphics.BIND_POINT_COMPUTE)
            ctx.cmd_bind_resources(2, [self.particles_pos_new], graphics.BIND_POINT_COMPUTE)
            ctx.cmd_bind_resources(3, [self.global_settings_buffer], graphics.BIND_POINT_COMPUTE)
            ctx.cmd_dispatch(self.n_particles // 128, 1, 1)
            ctx.cmd_copy_buffer(self.particles_pos, self.particles_pos_new, self.particles_pos.size())
        if self.renderer:
            self.renderer.record(ctx, self.particles_pos, self.n_particles)
        if self.window:
            ctx.cmd_present(self.window, self.renderer.frame_image)
        self.core.submit_command_context(ctx)
        if self.benchmark():
            self.core.wait_gpu()


def main(argv=None):
    try:
        options = parse_options(sys.argv if argv is None else argv)
        if options:
            NBodyCS(options).run()
        return 0
    except Exception as error:
        print(f"nbody_cs: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
