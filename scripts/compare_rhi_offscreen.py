"""Paired comparison of two Release rhix_hello binaries, including A/A controls.

Uses the demo's GPU-drained timing region, not process startup time. Run without
other builds/GPU workloads. Image readback validation must be performed separately.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import subprocess


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(values, rng):
    logs = [math.log(value) for value in values]
    boot = sorted(
        math.exp(statistics.mean(rng.choices(logs, k=len(logs))))
        for _ in range(10000)
    )
    return {
        "pairs": len(values),
        "geometric_mean_ratio": math.exp(statistics.mean(logs)),
        "paired_bootstrap_95_ci": [boot[249], boot[9749]],
        "min_ratio": min(values),
        "max_ratio": max(values),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=100000)
    parser.add_argument("--pairs", type=int, default=12)
    parser.add_argument("--control-pairs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=112)
    parser.add_argument("--backends", nargs="+", default=["vulkan", "d3d12"])
    parser.add_argument("--scenes", nargs="+", default=["cube", "texture", "ray_query"])
    args = parser.parse_args()
    if min(args.frames, args.pairs) < 1 or args.control_pairs < 0:
        parser.error("frames and comparison pairs must be positive; control pairs may be zero")
    args.baseline = args.baseline.resolve()
    args.candidate = args.candidate.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "baseline": str(args.baseline), "candidate": str(args.candidate),
        "baseline_sha256": fingerprint(args.baseline),
        "candidate_sha256": fingerprint(args.candidate),
        "frames": args.frames, "warmup_frames_per_process": 60,
        "max_in_flight": 3, "seed": args.seed,
        "pairs": args.pairs, "control_pairs": args.control_pairs,
        "backends": args.backends, "scenes": args.scenes,
        "metric": "candidate / baseline GPU-drained wall time; greater than 1 is slower",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    rng = random.Random(args.seed)
    cases = [(api, scene) for api in args.backends for scene in args.scenes]
    ratios = {}
    records = args.output / "samples.jsonl"

    def run(executable, api, scene, mode, pair, label, frames):
        command = [str(executable), "--backend", api, "--module", scene,
                   "--headless", "--frames", str(frames)]
        result = subprocess.run(command, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=180)
        log = result.stdout + result.stderr
        (args.output / f"{mode}-{api}-{scene}-{pair}-{label}.log").write_text(log, encoding="utf-8")
        match = re.search(rf"{scene}: headless 1280x720, {frames} frames, ([\d.]+) s,", log)
        if result.returncode or not match or "3 in flight, 60 warmup" not in log:
            raise RuntimeError(f"Failed measurement: {command}, return code {result.returncode}")
        seconds = float(match[1])
        with records.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"mode": mode, "api": api, "scene": scene,
                                     "pair": pair, "label": label, "frames": frames,
                                     "seconds": seconds}) + "\n")
        return seconds

    # Sustained untimed runs prime shader/driver caches and GPU clocks for both binaries.
    for api, scene in cases:
        for label, executable in [("A", args.baseline), ("B", args.candidate)]:
            run(executable, api, scene, "prime", 0, label, args.frames)
    for mode, count in [("control", args.control_pairs), ("comparison", args.pairs)]:
        orders = {}
        for case in cases:
            orders[case] = [index % 2 for index in range(count)]
            rng.shuffle(orders[case])
        for pair in range(count):
            rng.shuffle(cases)
            for api, scene in cases:
                order = ["A", "B"] if orders[(api, scene)][pair] == 0 else ["B", "A"]
                times = {}
                for label in order:
                    executable = args.candidate if mode == "comparison" and label == "B" else args.baseline
                    times[label] = run(executable, api, scene, mode, pair, label, args.frames)
                ratio = times["B"] / times["A"]
                ratios.setdefault((mode, api, scene), []).append(ratio)
                print(f"{mode} {pair + 1}/{count} {api}/{scene}: B/A={ratio:.5f}", flush=True)
    summary = {"/".join(key): summarize(values, rng) for key, values in ratios.items()}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
