"""Exercise the public AMR task interface and compare physical CSV outputs."""
import argparse
import csv
import ctypes
import math
from pathlib import Path
import subprocess
import tempfile


def run_task(binary, args):
    result = subprocess.run([str(binary), *args], capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError(f"Task failed ({result.returncode}): {args}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def compare(reference, actual):
    if len(reference) != len(actual):
        raise AssertionError("Different CSV row counts")
    maximum = 0.0
    for a, b in zip(reference, actual):
        for key in ("step", "time", "dt", "cells", "generation"):
            if a[key] != b[key]:
                raise AssertionError(f"Different {key}: {a} vs {b}")
        for key in ("Fx", "Fy", "Fz", "Mx", "My", "Mz"):
            expected, value = float(a[key]), float(b[key])
            if not math.isfinite(expected) or not math.isfinite(value):
                raise AssertionError(f"Non-finite load: {a} vs {b}")
            error = abs(value - expected)
            maximum = max(maximum, error)
            if error > 5e-7 + 2e-6 * abs(expected):
                raise AssertionError(f"Different {key}: {a} vs {b}")
    return maximum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("cuda_runtime")
    args = parser.parse_args()
    runtime = ctypes.CDLL(args.cuda_runtime)
    count = ctypes.c_int()
    if runtime.cudaGetDeviceCount(ctypes.byref(count)) or count.value < 2:
        return 77
    common = ["--grid", "64", "--steps", "8", "--warmup", "1", "--speed", "12", "--angle", "5",
              "--cubic", "--no-export", "--max-dt", ".0002"]
    for extra in (["--adaptive", "--gpus", "2"], ["--amr-uniform", "--gpus", "2"],
                  ["--adaptive", "--amr-compatible-pressure", "--gpus", "3"],
                  ["--adaptive", "--amr-compatible-pressure", "--gpus", "2", "--first-order"]):
        result = subprocess.run([str(args.binary), *common, *extra], capture_output=True, text=True, timeout=30)
        if result.returncode != 1:
            raise AssertionError(f"Expected rejected options: {extra}\n{result.stdout}\n{result.stderr}")
    devices = [1, 2] + ([4] if count.value >= 4 else [])
    with tempfile.TemporaryDirectory(prefix="amr-task-") as temporary:
        for mode in ("--amr-uniform", "--adaptive"):
            reference = None
            for number in devices:
                path = Path(temporary) / f"{mode[2:]}-{number}.csv"
                output = run_task(args.binary, [*common, mode, "--amr-compatible-pressure", "--gpus", str(number),
                                               "--adapt-every", "2", "--csv", str(path)])
                with path.open() as file:
                    rows = list(csv.DictReader(file))
                if len(rows) != 8 or f"gpus={number}" not in output or "RESULT backend=" not in output:
                    raise AssertionError(output)
                if mode == "--adaptive" and output.count("REGRID step=") != 3:
                    raise AssertionError(output)
                maximum = compare(reference or rows, rows)
                reference = reference or rows
                print(f"PASS task {mode} gpus={number} rows={len(rows)} maximum_load_difference={maximum:.9g}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
