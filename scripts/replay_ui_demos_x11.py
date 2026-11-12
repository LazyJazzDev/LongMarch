"""Replay keyboard, pointer and resize events against baseline and RHI on X11.

Run under a dedicated xvfb-run display. Requires xdotool and libX11. Input is
delivered through the real GLFW/X11 event path; rendering/presentation is enabled.
The final PNGs are intended for pixel comparison, not timing measurements.
"""

import argparse
import ctypes as c
import ctypes.util
import os
from pathlib import Path
import subprocess
import time


def xdo(*args):
    return subprocess.check_output(["xdotool", *map(str, args)], text=True, timeout=10).strip()


def request_close(window):
    # WM_DELETE_WINDOW asks GLFW to close normally, preserving screenshot export.
    class ClientMessage(c.Structure):
        _fields_ = [("type", c.c_int), ("serial", c.c_ulong), ("send_event", c.c_int),
                    ("display", c.c_void_p), ("window", c.c_ulong),
                    ("message_type", c.c_ulong), ("format", c.c_int), ("data", c.c_long * 5)]

    class Event(c.Union):
        _fields_ = [("client", ClientMessage), ("pad", c.c_long * 24)]

    lib = c.CDLL(ctypes.util.find_library("X11"))
    lib.XOpenDisplay.argtypes, lib.XOpenDisplay.restype = [c.c_char_p], c.c_void_p
    lib.XInternAtom.argtypes, lib.XInternAtom.restype = [c.c_void_p, c.c_char_p, c.c_int], c.c_ulong
    lib.XSendEvent.argtypes = [c.c_void_p, c.c_ulong, c.c_int, c.c_long, c.POINTER(Event)]
    lib.XFlush.argtypes = [c.c_void_p]
    lib.XCloseDisplay.argtypes = [c.c_void_p]
    display = lib.XOpenDisplay(None)
    if not display:
        raise RuntimeError("Cannot connect to X11")
    try:
        event = Event()
        event.client = ClientMessage(33, 0, 1, display, int(window),
                                     lib.XInternAtom(display, b"WM_PROTOCOLS", 0), 32,
                                     (c.c_long * 5)(lib.XInternAtom(display, b"WM_DELETE_WINDOW", 0), 0, 0, 0, 0))
        if not lib.XSendEvent(display, int(window), 0, 0, c.byref(event)):
            raise RuntimeError("Cannot request window close")
        lib.XFlush(display)
    finally:
        lib.XCloseDisplay(display)


def replay(executable, scenario, output, pattern):
    env = dict(os.environ, LONGMARCH_DEMO_SEED="1", LONGMARCH_WINDOW_SYSTEM="x11")
    env.pop("WAYLAND_DISPLAY", None)
    env.pop("LONGMARCH_DEMO_OFFSCREEN", None)
    env.pop("LONGMARCH_DEMO_BENCHMARK", None)
    command = [str(executable.resolve()), "--backend", "vulkan", "--screenshot", str(output.resolve())]
    if scenario.startswith("gol"):
        command += ["32", "32", "--pattern", str(pattern.resolve())]
    with output.with_suffix(".log").open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 15
            window = None
            while time.monotonic() < deadline and process.poll() is None:
                result = subprocess.run(["xdotool", "search", "--onlyvisible", "--pid", str(process.pid)],
                                        capture_output=True, text=True)
                expected = "2048 FPS:" if scenario.startswith("2048") else "Game of Life FPS:"
                candidates = []
                for candidate in result.stdout.split():
                    title = xdo("getwindowname", candidate)
                    if expected in title:
                        candidates.append(candidate)
                if len(candidates) == 1:
                    window = candidates[0]
                    break
                time.sleep(0.05)
            if window is None:
                raise RuntimeError(f"No unique demo window: {scenario}; last PID windows: {result.stdout!r}")
            xdo("windowfocus", "--sync", window)
            time.sleep(0.8)

            def key(value):
                xdo("key", "--window", window, "--clearmodifiers", value)
                time.sleep(0.35)

            def click(x, y):
                xdo("mousemove", "--window", window, x, y)
                time.sleep(0.05)
                xdo("click", "--window", window, 1)
                time.sleep(0.35)

            if scenario.startswith("2048"):
                for value in ("Left", "Down", "Right", "Up"):
                    key(value)
                if scenario == "2048-menu":
                    click(549, 170)
            else:
                if scenario == "gol-controls":
                    # The loaded 2x2 still life survives any number of timed steps.
                    click(72, 648)
                    time.sleep(0.6)
                    click(72, 648)
                    click(640, 360)
                    key("ctrl+plus")
                    key("ctrl+plus")
                    xdo("mousemove", "--window", window, 640, 360)
                    xdo("mousedown", "--window", window, 3)
                    xdo("mousemove", "--window", window, 720, 400)
                    xdo("mouseup", "--window", window, 3)
                else:
                    click(51, 300)
                    key("Right")
            if scenario.endswith("resize"):
                width, height = (960, 720) if scenario.startswith("2048") else (720, 960)
                xdo("windowsize", "--sync", window, width, height)
                time.sleep(0.4)
                xdo("windowunmap", "--sync", window)
                time.sleep(0.2)
                xdo("windowmap", "--sync", window)
                xdo("windowfocus", window)
            # Remove hover feedback and let all finite animations settle.
            xdo("mousemove", 1900, 1000)
            time.sleep(1.0)
            request_close(window)
            if process.wait(timeout=15) != 0 or not output.is_file():
                raise RuntimeError(f"Replay failed: {scenario}; see {log.name}")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pattern = args.output / "still-life.cells"
    pattern.write_text("! Stable block\nOO\nOO\n")
    for scenario in ("2048-moves", "2048-menu", "2048-resize", "gol-controls", "gol-resize"):
        demo = "2048" if scenario.startswith("2048") else "gol"
        for variant, root in (("baseline", args.baseline_root), ("rhi", args.candidate_root)):
            replay(root / demo / ("demo_" + demo), scenario,
                   args.output / f"{scenario}-{variant}.png", pattern)
            print(f"PASS {scenario} {variant}", flush=True)


if __name__ == "__main__":
    main()
