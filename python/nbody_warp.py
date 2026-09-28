"""NVIDIA Warp counterpart of demo/nbody_cuda.

A Warp kernel integrates the particles directly in a graphics vertex buffer shared with
CUDA (Core.create_cuda_buffer), so rendering needs no copies. CUDA work on Warp's stream
is ordered against graphics work with cuda_begin/end_execution_barrier, like
CUDABeginExecutionBarrier/CUDAEndExecutionBarrier in the C++ demo.

Requires a CUDA-enabled long_march build and Warp (pip install warp-lang); -headless also
writes final_frame.jpg with Pillow.
"""

import sys

import numpy as np
import warp as wp
from long_march import graphics

import nbody_common as nb

SHADER_DIRECTORY = nb.REPO_ROOT / "demo" / "nbody_cuda" / "shaders"
BLOCK_SIZE = 128
SOFTENING_SQUARED = 0.00125 * 0.00125

# The CUDA kernel uses the approximate rsqrt; fast math gives Warp the same instructions.
wp.set_module_options({"fast_math": True})


@wp.kernel
def update_kernel(positions: wp.array(dtype=wp.vec3), positions_write: wp.array(dtype=wp.vec3),
                  velocities: wp.array(dtype=wp.vec3), n_particle: int, delta_t: float, gravity: float):
    # One block per BLOCK_SIZE particles; each tile of positions is staged in shared memory,
    # like shared_pos in the CUDA kernel. n_particle must be a multiple of BLOCK_SIZE.
    block, lane = wp.tid()
    i = block * BLOCK_SIZE + lane
    pos = positions[i]
    vel = velocities[i]
    accel = wp.vec3(0.0)
    for tile_index in range(n_particle // BLOCK_SIZE):
        tile = wp.tile_load(positions, shape=BLOCK_SIZE, offset=tile_index * BLOCK_SIZE, storage="shared")
        for k in range(wp.static(BLOCK_SIZE)):
            diff = pos - tile[k]
            l = 1.0 / wp.sqrt(wp.dot(diff, diff) + SOFTENING_SQUARED)
            accel += diff * (l * l * l * (-delta_t * gravity))
    vel += accel
    positions_write[i] = pos + vel * delta_t
    velocities[i] = vel


def print_arg_helper(command_name):
    print(f"Usage: {command_name} [-headless] [-nstep <num_steps>] [-device <device_id>]")
    print("Options:")
    print("  -headless           Run in headless mode (no GUI)")
    print("  -nstep <num_steps>  Number of simulation steps to run (default: 200)")
    print("  -device <device_id> CUDA device ID to use (default: 0)")


class NBodyWarp:
    def __init__(self, n_particles=nb.NUM_PARTICLE, headless=False, num_step=200, device_id=0):
        self.n_particles = n_particles
        self.headless = headless
        self.num_step = num_step
        self.rng = np.random.default_rng()
        self.core = graphics.Core(graphics.BACKEND_API_DEFAULT)
        self.core.init_cuda(device_id)
        self.window = None
        if not headless:
            self.window = self.core.create_window(1920, 1080, "NBody Warp", False, True)
        self.device = None
        cuda_device_index = self.core.cuda_device_index()
        if cuda_device_index >= 0:
            self.device = wp.get_device(f"cuda:{cuda_device_index}")
            nb.log_info(f'GPU Device {cuda_device_index}: "{self.device.name}" with compute capability '
                        f"{self.device.arch // 10}.{self.device.arch % 10}")
            self.stream = wp.get_stream(self.device)
        else:
            print("[error] Selected graphics device is not a CUDA device.", file=sys.stderr)
        self.controls = None
        self.fps_counter = nb.FPSCounter()

    def run(self):
        if self.device is None:
            return
        if self.headless:
            nb.log_info(f"Headless mode enabled. Running simulation for {self.num_step} steps...")
            self.on_init()
            fps_counter = nb.FPSCounter()
            progress = 0
            for step in range(self.num_step):
                self.update_particles()
                fps_counter.tick_frame()
                if step * 10 // self.num_step > progress:
                    progress = step * 10 // self.num_step
                    nb.log_info(f"Progress: {progress * 10}%... ({step} / {self.num_step})")
            nb.log_info("Done.")
            self.update_render_assets()
            self.on_render()
            self.save_frame("final_frame.jpg")
            nb.log_info(f"Frames per second: {fps_counter.get_fps():.2f}")
            nb.log_info("Final frame saved to final_frame.jpg")
            self.core.wait_gpu()
            self.on_close()
        else:
            self.on_init()
            while not self.window.should_close():
                self.on_update()
                self.on_render()
                graphics.Window.poll_events()
            self.core.wait_gpu()
            self.on_close()

    def save_frame(self, path):
        from PIL import Image  # Only the headless snapshot needs Pillow.

        extent = self.renderer.frame_image.extent()
        encoded = np.frombuffer(self.renderer.frame_image.download_data(), np.float32)
        pixels = (np.clip(encoded, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8).reshape(extent.height, extent.width, 4)
        Image.fromarray(pixels[:, :, :3]).save(path, quality=100)

    def on_init(self):
        if not self.headless:
            self.renderer = nb.ParticleRenderer(self.core, SHADER_DIRECTORY, *self.window.get_framebuffer_size())
            self.window.register_framebuffer_resize_event(self.renderer.resize)
            self.window.init_imgui(str(nb.REPO_ROOT / "assets" / "fonts" / "simhei.ttf"), 20.0)
            self.controls = nb.NBodyControls(self.window, "NBody Warp")
        else:
            self.renderer = nb.ParticleRenderer(self.core, SHADER_DIRECTORY, 960, 640)
        nb.log_info(f"Simulating {self.n_particles} particles...")
        self.particles_pos = self.core.create_cuda_buffer(12 * self.n_particles)
        self.positions = wp.array(ptr=self.particles_pos.cuda_ptr(), dtype=wp.vec3, shape=self.n_particles,
                                  device=self.device)
        self.velocities = wp.empty(self.n_particles, dtype=wp.vec3, device=self.device)
        self.positions_new = wp.empty(self.n_particles, dtype=wp.vec3, device=self.device)
        self.reset_particles()

    def on_close(self):
        del self.positions, self.velocities, self.positions_new
        del self.renderer, self.particles_pos

    def on_update(self):
        self.update_particles()
        if not self.headless:
            self.controls.update_imgui(self.n_particles, self.download_velocities, self.reset_particles)
            self.update_render_assets()
            self.window.set_title(f"NBody Warp FPS: {self.fps_counter.tick_fps():.6f}")

    def on_render(self):
        ctx = self.core.create_command_context()
        self.renderer.record(ctx, self.particles_pos, self.n_particles)
        if not self.headless:
            ctx.cmd_present(self.window, self.renderer.frame_image)
        self.core.submit_command_context(ctx)

    def galaxy_number(self):
        return self.controls.galaxy_number if self.controls else 10

    def delta_t(self):
        return self.controls.delta_t if self.controls else nb.DELTA_T

    def reset_particles(self):
        positions, velocities = nb.initial_particles(self.rng, self.n_particles, self.galaxy_number())
        self.core.cuda_begin_execution_barrier(self.stream.cuda_stream)
        with wp.ScopedStream(self.stream):
            self.positions.assign(positions)
            self.velocities.assign(velocities)
        self.core.cuda_end_execution_barrier(self.stream.cuda_stream)

    def download_velocities(self):
        self.core.cuda_begin_execution_barrier(self.stream.cuda_stream)
        with wp.ScopedStream(self.stream):
            velocities = self.velocities.numpy()
        self.core.cuda_end_execution_barrier(self.stream.cuda_stream)
        return velocities

    def update_particles(self):
        if self.controls and not self.controls.step:
            return
        self.core.cuda_begin_execution_barrier(self.stream.cuda_stream)
        wp.launch_tiled(update_kernel, dim=[self.n_particles // BLOCK_SIZE], block_dim=BLOCK_SIZE, stream=self.stream,
                        inputs=[self.positions, self.positions_new, self.velocities, self.n_particles, self.delta_t(),
                                nb.GRAVITY_COE])
        wp.copy(self.positions, self.positions_new, stream=self.stream)
        self.core.cuda_end_execution_barrier(self.stream.cuda_stream)

    def update_render_assets(self):
        rotation = self.controls.rotation if self.controls else np.identity(4, np.float32)
        self.renderer.update_uniforms(rotation, self.controls.hdr if self.controls else False)


def main(argv=None):
    argv = sys.argv if argv is None else argv
    settings = {}
    head = 1
    while head < len(argv):
        arg = argv[head]
        if arg == "-headless":
            settings["headless"] = True
        elif arg in ("-nstep", "-device"):
            if head + 1 >= len(argv):
                print_arg_helper(argv[0])
                return 1
            head += 1
            settings["num_step" if arg == "-nstep" else "device_id"] = int(argv[head])
        else:
            print_arg_helper(argv[0])
            return 1
        head += 1
    wp.config.log_level = wp.LOG_WARNING
    wp.init()
    NBodyWarp(**settings).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
