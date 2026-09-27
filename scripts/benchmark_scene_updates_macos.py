"""Run the PR64 Metal comparison or summarize its retained frame data."""

import argparse
import csv
from pathlib import Path
import statistics
import subprocess

SCENES = {"classroom": (1920, 1080), "junkshop": (2000, 1000), "monster": (1024, 1024)}
SCOPES = ("frame_wall", "scene_update", "scene_registration", "software_prepare", "render_wait")
ROOT = Path(__file__).resolve().parents[1]


def collect(directory, destination):
    rows = []
    for run in range(1, 4):
        for scene in SCENES:
            for phase in ("before", "after"):
                frames = {}
                with (directory / f"{scene}-{phase}-{run}.csv").open() as source:
                    for row in csv.DictReader(source):
                        if row["domain"] == "cpu_ms" and row["stage"] in SCOPES:
                            frames.setdefault(int(row["frame"]), {})[row["stage"]] = row["value"]
                if set(frames) != set(range(60)):
                    raise ValueError(f"Incomplete run: {scene}-{phase}-{run}")
                for frame in range(10, 60):
                    rows.append([phase, scene, "metal", run, frame] +
                                [frames[frame][scope] for scope in SCOPES])
            before = directory / f"{scene}-before-{run}.png"
            after = directory / f"{scene}-after-{run}.png"
            if before.read_bytes() != after.read_bytes():
                raise ValueError(f"PNG mismatch: {scene}, run {run}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["phase", "scene", "backend", "run", "frame"] + list(SCOPES))
        writer.writerows(rows)


def summarize(path):
    with path.open() as source:
        rows = list(csv.DictReader(source))
    print("scene,before_ms,after_ms,before_M_samples_s,after_M_samples_s,throughput_change_percent")
    for scene, (width, height) in SCENES.items():
        means = []
        for phase in ("before", "after"):
            selected = [row for row in rows if row["scene"] == scene and row["phase"] == phase]
            if (len(selected) != 150 or
                    {(int(r["run"]), int(r["frame"])) for r in selected} !=
                    {(run, frame) for run in range(1, 4) for frame in range(10, 60)}):
                raise ValueError(f"Expected 150 retained frames: {scene}/{phase}")
            means.append(statistics.mean(float(row["frame_wall"]) for row in selected))
        before, after = means
        samples = width * height * 8 / 1000
        print(f"{scene},{before:.3f},{after:.3f},{samples / before:.3f},"
              f"{samples / after:.3f},{(before / after - 1) * 100:+.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, help="Baseline CLI executable")
    parser.add_argument("--after", type=Path, help="PR CLI executable")
    parser.add_argument("--output", type=Path, default=ROOT / "out/scene-updates-macos")
    parser.add_argument("--collect", type=Path, help="Collect existing raw logs and compare PNGs")
    parser.add_argument("--csv", type=Path, default=ROOT / "docs/reports/nsight-scene-update-macos-frames.csv")
    args = parser.parse_args()
    if bool(args.before) != bool(args.after):
        parser.error("--before and --after must be supplied together")
    if args.before and args.collect:
        parser.error("Choose a new benchmark or --collect")
    if args.before:
        # A fresh directory prevents mixing or overwriting independent measurements.
        args.output.mkdir(parents=True, exist_ok=False)
        binaries = {"before": args.before.resolve(), "after": args.after.resolve()}
        for run in range(1, 4):
            for scene in SCENES:
                for phase in (("before", "after") if run % 2 else ("after", "before")):
                    stem = args.output.resolve() / f"{scene}-{phase}-{run}"
                    command = [str(binaries[phase]), str(ROOT / f"assets/scenes/blender_{scene}/scene.json"),
                               "--backend", "metal", "--pipeline", "ray_query", "--frames", "60",
                               "--profile-cpu-only", "--profile", str(stem) + ".csv", "-o", str(stem) + ".png"]
                    print(stem.name, flush=True)
                    with Path(str(stem) + ".log").open("w") as log:
                        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        args.collect = args.output
    if args.collect:
        # New measurements stay with their raw artifacts, leaving published data intact.
        args.csv = args.collect / "frames.csv"
        collect(args.collect, args.csv)
    summarize(args.csv)


if __name__ == "__main__":
    main()
