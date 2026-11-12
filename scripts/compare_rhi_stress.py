"""Reproducible, paired end-to-end offscreen stress measurements.

Prepare validates raw output and calibrates run lengths. Measure executes two
independently randomized blocks of eight A/B pairs plus two A/A pairs per case.
Use --performance-cores on Windows to avoid heterogeneous-core migration; it
changes only this process and its children, not system settings.
"""
import argparse
import ctypes
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import subprocess
import sys
import time

PROFILES = {
    "draw": {"grid": 64, "width": 1280, "height": 720, "batch": 256, "iterations": 1},
    "shader": {"grid": 16, "width": 3840, "height": 2160, "batch": 64, "iterations": 2048},
}


def affinity(performance_cores):
    if not performance_cores:
        return None
    if sys.platform != "win32":
        raise RuntimeError("--performance-cores requires Windows")
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GetSystemCpuSetInformation.argtypes = [ctypes.c_void_p, ctypes.c_ulong,
                                            ctypes.POINTER(ctypes.c_ulong), ctypes.c_void_p, ctypes.c_ulong]
    size = ctypes.c_ulong()
    k.GetSystemCpuSetInformation(None, 0, ctypes.byref(size), None, 0)
    data = ctypes.create_string_buffer(size.value)
    if not k.GetSystemCpuSetInformation(data, size, ctypes.byref(size), None, 0):
        raise ctypes.WinError(ctypes.get_last_error())
    cpus = []
    offset = 0
    while offset < size.value:
        length, kind = struct.unpack_from("<II", data.raw, offset)
        if length < 8:
            raise RuntimeError("Invalid CPU-set entry")
        if kind == 0:
            _, group, logical, core, _, _, efficiency, flags = struct.unpack_from("<IHBBBBBB", data.raw, offset + 8)
            cpus.append({"group": group, "logical": logical, "core": core,
                         "efficiency": efficiency, "flags": flags})
        offset += length
    if not cpus or any(c["group"] != 0 for c in cpus):
        raise RuntimeError("This runner only supports one processor group")
    fastest = max(c["efficiency"] for c in cpus)
    selected = [c["logical"] for c in cpus if c["efficiency"] == fastest]
    mask = sum(1 << index for index in selected)
    k.GetCurrentProcess.restype = ctypes.c_void_p
    k.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    if not k.SetProcessAffinityMask(k.GetCurrentProcess(), mask):
        raise ctypes.WinError(ctypes.get_last_error())
    return {"logical_processors": selected, "mask": mask, "topology": cpus}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def telemetry():
    result = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.gr,clocks.sm,power.draw,utilization.gpu",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
    return result.stdout.strip() if result.returncode == 0 else None


def run(executable, api, profile, frames, warmup, directory, label, capture=False):
    command = [str(executable), "--backend", api, "--frames", str(frames), "--warmup", str(warmup)]
    for name, value in profile.items():
        command.extend(["--" + name, str(value)])
    if capture:
        command.extend(["--capture", str(directory / label)])
    begin = time.time()
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    log = result.stdout + result.stderr
    (directory / (label + ".log")).write_text(log, encoding="utf-8")
    records = [json.loads(line[7:]) for line in result.stdout.splitlines() if line.startswith("RESULT ")]
    if result.returncode or len(records) != 1:
        raise RuntimeError(f"Measurement failed ({result.returncode}): {command}; see {label}.log")
    record = records[0]
    for key, value in {**profile, "frames": frames, "warmup": warmup, "in_flight": 3}.items():
        if key == "batch":
            continue
        if record.get(key) != value:
            raise RuntimeError(f"Unexpected workload: {key}: {record.get(key)} != {value}")
    if record["draws_per_frame"] != profile["grid"] ** 2:
        raise RuntimeError("Incorrect draw count")
    if record["submissions_per_frame"] != math.ceil(profile["grid"] ** 2 / profile["batch"]):
        raise RuntimeError("Incorrect submission count")
    if not all(math.isfinite(record[k]) and record[k] > 0 for k in ("seconds", "record_seconds")):
        raise RuntimeError("Invalid timing")
    return {**record, "started_unix": begin, "process_seconds": time.time() - begin, "command": command}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "measure"])
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--performance-cores", action="store_true")
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--seed", type=int, default=1142026)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be finite and positive")
    executables = {"A": args.baseline.resolve(), "B": args.candidate.resolve()}
    hashes = {name: sha(path) for name, path in executables.items()}
    mask = affinity(args.performance_cores)
    cases = [(api, profile) for api in ["vulkan", "d3d12"] for profile in PROFILES]
    if args.mode == "prepare":
        args.output.mkdir(parents=True, exist_ok=False)
        plan = {"profiles": PROFILES, "hashes": hashes, "affinity": mask, "cases": {},
                "seed": args.seed, "target_seconds": args.seconds, "blocks": 2,
                "comparison_pairs_per_block": 8, "control_pairs_per_block": 2,
                "warmup": 120, "practical_equivalence_percent": 3,
                "note": "All valid samples retained; two independently balanced blocks; A/A interleaved with A/B."}
        for api, name in cases:
            key = api + "/" + name
            profile = PROFILES[name]
            rows = []
            for label, exe in executables.items():
                capture_name = f"capture-{api}-{name}-{label}"
                run(exe, api, profile, 8, 2, args.output, capture_name, True)
                row = run(exe, api, profile, 200, 120, args.output, f"calibrate-{api}-{name}-{label}")
                rows.append(row)
            a = args.output / f"capture-{api}-{name}-A/stress.bin"
            b = args.output / f"capture-{api}-{name}-B/stress.bin"
            if a.read_bytes() != b.read_bytes():
                raise RuntimeError(f"Raw image mismatch: {key}")
            pixels = a.read_bytes()
            if len(pixels) != profile["width"] * profile["height"] * 4 or len(set(pixels)) < 64:
                raise RuntimeError(f"Invalid/empty image: {key}")
            mean_duration = sum(row["seconds"] / row["frames"] for row in rows) / len(rows)
            frames = max(200, math.ceil(args.seconds / mean_duration / 50) * 50)
            plan["cases"][key] = {"frames": frames, "capture_sha256": sha(a), "calibration": rows}
            print(f"PREPARE {key}: exact image match; {frames} measured frames", flush=True)
        # Verify that the shader profile really scales with arithmetic work.
        plan["shader_scaling"] = []
        for api in ["vulkan", "d3d12"]:
            for label, exe in executables.items():
                row = run(exe, api, {**PROFILES["shader"], "iterations": 256}, 200, 120,
                          args.output, f"scaling-{api}-{label}")
                plan["shader_scaling"].append({"api": api, "label": label, **row})
        (args.output / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
        return
    plan = json.loads((args.output / "plan.json").read_text(encoding="utf-8"))
    if plan["hashes"] != hashes or (plan["affinity"] or {}).get("mask") != (mask or {}).get("mask"):
        raise RuntimeError("Binaries or CPU affinity differ from preparation")
    destination = args.output / "samples.jsonl"
    if destination.exists():
        raise RuntimeError("Refusing to overwrite an existing run")
    rng = random.Random(plan["seed"])
    # Long priming runs for both binaries before recording paired samples.
    for api, name in cases:
        for label, exe in executables.items():
            run(exe, api, PROFILES[name], plan["cases"][api + "/" + name]["frames"],
                plan["warmup"], args.output, f"prime-{api}-{name}-{label}")
    for block in range(plan["blocks"]):
        modes = ["control"] + ["comparison"] * 4 + ["control"] + ["comparison"] * 4
        orders = {}
        for case in cases:
            for mode, count in [("control", 2), ("comparison", 8)]:
                seq = [n % 2 for n in range(count)]
                rng.shuffle(seq)
                orders[case, mode] = iter(seq)
        indices = {"control": 0, "comparison": 0}
        for mode in modes:
            pair = indices[mode]
            indices[mode] += 1
            rng.shuffle(cases)
            for api, name in cases:
                values = {}
                order = ["A", "B"] if next(orders[(api, name), mode]) == 0 else ["B", "A"]
                for label in order:
                    actual = "A" if mode == "control" else label
                    tag = f"{block}-{mode}-{pair}-{api}-{name}-{label}"
                    before = telemetry()
                    row = run(executables[actual], api, PROFILES[name],
                              plan["cases"][api + "/" + name]["frames"], plan["warmup"], args.output, tag)
                    row.update({"block": block, "mode": mode, "pair": pair, "api": api, "profile": name,
                                "label": label, "actual_binary": actual, "order": order,
                                "gpu_before": before, "gpu_after": telemetry()})
                    with destination.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row) + "\n")
                    values[label] = row["seconds"]
                print(f"BLOCK {block + 1}/2 {mode} {pair + 1}: {api}/{name} B/A={values['B']/values['A']:.5f}", flush=True)
    (args.output / "complete.json").write_text(json.dumps({"completed_unix": time.time(), "hashes": hashes}), encoding="utf-8")


if __name__ == "__main__":
    main()
