#!/usr/bin/env python3
"""Run identical offscreen workloads in balanced, randomized revision order."""
import argparse
import csv
import json
import random
import subprocess
import time
from pathlib import Path

CASES = [
    ("upload-small", "upload", 65536, 8, 32768),
    ("upload-compute", "upload", 1048576, 256, 16384),
    ("as-build", "as-build", 256, 8, 4096),
    ("as-update", "as-update", 256, 8, 8192),
]
FIELDS = ["revision", "api", "mode", "workload", "size", "loops", "depth", "frames", "seconds", "fps", "mismatches", "device"]


def gpu_state():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu", "--format=csv,noheader"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10,
    )
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--apis", default="vulkan,d3d12")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--depth", type=int, default=8)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}")
    rng = random.Random(113)
    with output.open("w", encoding="utf-8", newline="") as csv_file, output.with_suffix(".jsonl").open("w", encoding="utf-8") as log:
        writer = csv.DictWriter(csv_file, fieldnames=["round", "case"] + FIELDS)
        writer.writeheader()
        for round_index in range(args.rounds):
            cases = [(api, case) for api in args.apis.split(",") for case in CASES]
            rng.shuffle(cases)
            for api, (label, workload, size, loops, frames) in cases:
                modes = [(args.baseline, "main"), (args.current, "main"), (args.current, "assist")]
                rng.shuffle(modes)
                for executable, mode in modes:
                    command = [executable, api, mode, workload, str(size), str(loops), str(args.depth), str(frames), "0"]
                    before = gpu_state()
                    begin = time.time()
                    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
                    record = dict(command=command, start=begin, elapsed=time.time()-begin, gpu_before=before,
                                  gpu_after=gpu_state(), returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
                    log.write(json.dumps(record, ensure_ascii=False)+"\n")
                    log.flush()
                    lines = [line for line in result.stdout.splitlines() if line.startswith("RESULT,")]
                    if result.returncode or len(lines) != 1:
                        raise RuntimeError(f"Failed benchmark: {command}\n{result.stdout}\n{result.stderr}")
                    row = dict(zip(FIELDS, next(csv.reader([lines[0]]))[1:], strict=True))
                    if int(row["mismatches"]):
                        raise RuntimeError("Output verification failed")
                    writer.writerow(dict(round=round_index, case=label, **row))
                    csv_file.flush()
                    print(f'{round_index} {api} {label} {row["revision"]}/{mode}: {row["fps"]} fps', flush=True)


if __name__ == "__main__":
    main()
