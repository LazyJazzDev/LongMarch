"""Run headless scene/CLI regressions. Usage: python test_headless_demos.py <demo> [<demo> ...]"""
import argparse
from pathlib import Path
import re
import subprocess
import sys


def run(executable, args, success=True):
    result = subprocess.run([str(executable), *args], capture_output=True, timeout=90)
    output = (result.stdout + result.stderr).decode("utf-8", errors="replace")
    if (result.returncode == 0) != success:
        raise RuntimeError(f"{executable.name} {args}: exit {result.returncode}\n{output}")
    if re.search(r"Validation Error|VUID-|SYNC-HAZARD", output):
        raise RuntimeError(output)
    return output


def check(executable, backend):
    for scene in ("hello_window", "hello_triangle", "hello_cube", "hello_texture", "all"):
        selection = ["--all"] if scene == "all" else ["--module", scene]
        output = run(executable, ["--headless", "--backend", backend, "--frames", "7", *selection])
        rows = re.findall(r"(hello_\w+): headless 1280x720, (\d+) frames, ([\d.]+) s, ([\d.]+) FPS", output)
        expected = ["hello_window", "hello_triangle", "hello_cube", "hello_texture"] if scene == "all" else [scene]
        if [row[0] for row in rows] != expected:
            raise RuntimeError(f"Missing or repeated scene results:\n{output}")
        for _, frames, seconds, fps in rows:
            if int(frames) != 7 or float(seconds) <= 0 or float(fps) <= 0:
                raise RuntimeError(f"Invalid timing results:\n{output}")
    output = run(executable, ["--headless", "--all", "--backend", "d3d12" if sys.platform == "darwin" else "metal", "--frames", "1"], success=False)
    if "hello_window: FAILED" not in output or "hello_triangle: FAILED" not in output:
        raise RuntimeError(f"--all did not continue after failure:\n{output}")
    for args in (["--all", "--module", "hello_window"], ["--all", "--tui"],
                 ["--headless", "--all", "--frames", "-1"], ["--frames", "invalid"]):
        run(executable, args, success=False)
    print(f"{executable.name}/{backend}: headless scenes, --all and invalid options passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executables", nargs="+", type=Path)
    parser.add_argument("--backend", choices=["d3d12", "vulkan", "metal"], required=True)
    args = parser.parse_args()
    for executable in args.executables:
        check(executable.resolve(), args.backend)
