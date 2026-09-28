"""Python counterpart of demo/graphics_hello: one launcher, eleven modules.

Each module lives in graphics_hello_<name>.py, mirrors demo/graphics_hello/modules/<name>,
and compiles the same shader sources. The command line matches demo_graphics_hello.
"""

import importlib
import math
import pathlib
import sys
import time

import numpy as np
from long_march import graphics

from glm_math import as_bytes, look_at, perspective, rotate_y, scale, translate  # noqa: F401 (re-exported)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODULE_ROOT = REPO_ROOT / "demo" / "graphics_hello"

MODULES = (
    ("triangle", "Colored triangle"),
    ("blend", "Alpha blending"),
    ("cube", "Rotating cube"),
    ("texture", "Textured triangle"),
    ("resize", "Resizable window"),
    ("hdr", "HDR gradient and SDR reference (H toggles HDR/SDR)"),
    ("sdr_sample", "SDR sampling"),
    ("raytracing", "Ray tracing pipeline (unavailable on Metal)"),
    ("rt_multi_shader_group", "Triangle + procedural sphere (requires RT pipelines)"),
    ("external_shader", "RT scene with shaders loaded from assets (requires RT pipelines)"),
    ("ray_query", "Compute ray queries (requires device/backend support)"),
)

BACKENDS = {
    "auto": graphics.BACKEND_API_DEFAULT,
    "metal": graphics.BACKEND_API_METAL,
    "vulkan": graphics.BACKEND_API_VULKAN,
    "d3d12": graphics.BACKEND_API_D3D12,
}

GLFW_PRESS = 1
GLFW_KEY_H = 72


def log_info(message):
    print(f"[info] {message}", flush=True)


def backend_name(api):
    return graphics.backend_api_string(api)


def graphics_hello_title(api):
    return f"[{backend_name(api)}]"


def load_shader(path):
    """Read a shader shared with the C++ launcher, e.g. "modules/triangle/shaders/shader.slang"."""
    return (MODULE_ROOT / path).read_text()


def initialize_graphics_hello(api, require_ray_tracing=False):
    if not graphics.support_backend_api(api):
        raise RuntimeError("Requested graphics backend is unavailable")
    core = graphics.Core(api)
    if core.init_auto(require_ray_tracing) != 0:
        raise RuntimeError("No compatible graphics device found")
    log_info(f"Backend API: {backend_name(core.api())}")
    log_info(f"Device Name: {core.device_name()}")
    log_info(f"- Ray Tracing Support: {core.ray_tracing_support()}")
    log_info(f"- Ray Query Support: {core.ray_query_support()}")
    return core


def instance_transform(m):
    """Row-major 3x4 transform accepted by AccelerationStructure.make_instance."""
    return m[:3].tolist()


def camera_object(window, eye):
    """CameraObject {screen_to_camera, camera_to_world} used by the ray tracing modules."""
    proj = perspective(math.radians(60.0), window.get_width() / window.get_height(), 0.1, 10.0)
    view = look_at(eye, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    return as_bytes(np.linalg.inv(proj), np.linalg.inv(view))


class Module:
    """Lifecycle shared with graphics_hello::Module."""

    def __init__(self):
        self.window = None
        self.alive = False
        self._animation_start = None

    def on_init(self):
        raise NotImplementedError

    def on_close(self):
        raise NotImplementedError

    def on_update(self):
        raise NotImplementedError

    def on_render(self):
        raise NotImplementedError

    def update_alive(self):
        if self.window.should_close():
            self.alive = False
        return self.alive

    def rotation_angle(self):
        now = time.monotonic()
        if self._animation_start is None:
            self._animation_start = now
        # Start at the first animation update, after loading. One revolution takes two seconds.
        return math.fmod(now - self._animation_start, 2.0) * math.pi


def find_module(name):
    for module_name, _ in MODULES:
        if name == module_name:
            return module_name
    return None


def create_module(name, api):
    return importlib.import_module(f"graphics_hello_{name}").Module(api)


def positive_integer(value):
    if not value.isdigit() or int(value) <= 0 or int(value) > 2**31 - 1:
        raise ValueError(f"Expected a positive integer: {value}")
    return int(value)


def parse_backend(name):
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend: {name}")
    return BACKENDS[name]


def list_modules():
    for i, (name, description) in enumerate(MODULES):
        print(f"  {i + 1}. {name:<21} {description}")


# A line-based TUI works in native terminals, IDE consoles and redirected input.
def select_module(api):
    print(f"\nGraphics Hello - Module Selection\nBackend API: {backend_name(api)}\n")
    list_modules()
    while True:
        print(f"\nSelect module [1-{len(MODULES)} or name], q to quit: ", end="", flush=True)
        line = sys.stdin.readline()
        if not line:
            return None
        choice = line.strip(" \t\r\n")
        if not choice:
            continue
        if choice in ("q", "quit"):
            return None
        if find_module(choice):
            return choice
        if choice.isdigit() and 1 <= int(choice) <= len(MODULES):
            return MODULES[int(choice) - 1][0]
        print("Invalid selection. Enter a listed number or module name.")


def run_module(name, api, frames):
    log_info(f"Module: {name}")
    module = create_module(name, api)
    module.on_init()
    window = module.window
    title = window.get_title()
    window.set_title(title + " | FPS: --")
    fps_start = time.monotonic()
    fps_frames = 0
    rendered = 0
    while module.alive and (not frames or rendered < frames):
        graphics.Window.poll_events()
        module.on_update()
        if module.alive:
            module.on_render()
            if frames:
                rendered += 1
            fps_frames += 1
            now = time.monotonic()
            seconds = now - fps_start
            if seconds >= 0.5:
                window.set_title(f"{title} | FPS: {fps_frames / seconds:.1f}")
                fps_start = now
                fps_frames = 0
    module.on_close()


def print_help(executable):
    print(f"Usage: {executable} [--module NAME | --tui] [--backend auto|metal|vulkan|d3d12] [--frames N]\n"
          "       --list     List all modules without opening a window\n"
          "       --help     Show this help\n"
          "With no module specified, the terminal selection menu opens.\n")
    list_modules()


def main(argv=None, module=None):
    """Run the launcher. A module script passes its own name as the preselected module."""
    argv = sys.argv if argv is None else argv
    try:
        api = graphics.BACKEND_API_DEFAULT
        frames = 0
        tui = False
        show_list = False
        i = 1
        while i < len(argv):
            option = argv[i]
            if option == "--help":
                print_help(argv[0])
                return 0
            if option == "--tui" and module is None:
                tui = True
            elif option == "--list" and module is None:
                show_list = True
            elif option in ("--module", "--backend", "--frames") and i + 1 < len(argv):
                i += 1
                value = argv[i]
                if option == "--module" and module is None:
                    module = find_module(value)
                    if not module:
                        raise ValueError(f"Unknown module: {value}; use --list to see available modules")
                elif option == "--backend":
                    api = parse_backend(value)
                elif option == "--frames":
                    frames = positive_integer(value)
                else:
                    raise ValueError(f"Unknown or incomplete option: {option}")
            else:
                raise ValueError(f"Unknown or incomplete option: {option}")
            i += 1
        if tui and module:
            raise ValueError("Use either --module or --tui, not both")
        if show_list:
            list_modules()
            return 0
        if not graphics.support_backend_api(api):
            raise RuntimeError("Requested graphics backend is unavailable")
        if not module:
            module = select_module(api)
        if module:
            run_module(module, api, frames)
        return 0
    except Exception as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
