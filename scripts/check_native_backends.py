#!/usr/bin/env python3
"""Numerical CPU/CUDA/Vulkan path-tracer regression, without modifying pinned assets.

Requires numpy and Pillow. Saves input JSON, commands, logs, PNG/PFM, amplified
image differences and machine-readable metrics. A missing/failing backend is a
failure, never a successful skip. The existing deterministic Sobol sequence starts
at sample zero on every fresh process; there is no wall-clock seed.
"""
import argparse
import datetime
import itertools
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import time

import numpy as np
from PIL import Image
from check_rt_fallback import ROOT, DEMOS, read_scene, graph_smoke


def hair_scene(directory):
    # Deliberately exercise the same SPKHAIR1/SPKMESH1 loaders as production scenes.
    points = [
        (x + 0.15 * math.sin(y * 3), y, 0)
        for x in (-0.8, 0, 0.8)
        for y in (-1, -0.5, 0, 0.5, 1)
    ]
    radii = [0.12 - (i % 5) * 0.015 for i in range(15)]
    hair = directory / "fixture.hair"
    hair.write_bytes(
        b"SPKHAIR1"
        + struct.pack("<II4I", 3, 15, 0, 5, 10, 15)
        + b"".join(struct.pack("<3f", *p) for p in points)
        + struct.pack("<15f", *radii)
    )
    mesh = directory / "fixture.mesh"
    mesh.write_bytes(
        b"SPKMESH1"
        + struct.pack("<III6I", 4, 6, 7, 0, 1, 2, 0, 2, 3)
        + struct.pack(
            "<12f", -2, -1.3, -0.4, 2, -1.3, -0.4, 2, 1.3, -0.4, -2, 1.3, -0.4
        )
        + struct.pack("<12f", *(0, 0, 1) * 4)
        + struct.pack("<8f", 0, 0, 1, 0, 1, 1, 0, 1)
        + struct.pack("<12f", 1, 0.1, 0.1, 0.1, 1, 0.1, 0.1, 0.1, 1, 1, 1, 0.1)
    )
    return {
        "format": "sparkium-scene",
        "version": 1,
        "name": "hair tubes and vertex colors",
        "film": {"width": 64, "height": 64, "view_transform": "standard"},
        "renderer": {
            "samples_per_dispatch": 8,
            "max_bounces": 8,
            "background_color": [0.02, 0.03, 0.04],
        },
        "camera": {"eye": [0, 0, 5], "target": [0, 0, 0], "fov_degrees": 45},
        "materials": {
            "hair": {
                "type": "principled",
                "base_color": [0.55, 0.16, 0.035],
                "roughness": 0.25,
                "anisotropic": 0.6,
                "sheen": 0.5,
                "clearcoat": 0.3,
            },
            "color": {
                "type": "shader_graph",
                "graph": {
                    "nodes": {"v": {"type": "vertex_attribute"}},
                    "surface": {
                        "base_color": {"node": "v", "output": "color"},
                        "roughness": 0.7,
                    },
                },
            },
        },
        "geometries": {
            "hair": {"type": "hair", "path": str(hair), "radial_segments": 5},
            "quad": {
                "type": "binary_mesh",
                "path": str(mesh),
                "generate_tangents": True,
            },
        },
        "entities": [
            {"type": "mesh", "geometry": "quad", "material": "color"},
            {
                "type": "mesh",
                "geometry": "hair",
                "material": "hair",
                "transform": {"scale": [-1, 1.05, 0.8], "rotation_degrees": [0, 10, 0]},
            },
            {
                "type": "point_light",
                "position": [-2, 3, 4],
                "color": [1, 0.9, 0.8],
                "strength": 90,
                "radius": 0.5,
                "soft_falloff": True,
            },
        ],
    }


def fixture(name, directory):
    if name == "graph_smoke":
        return graph_smoke()
    if name == "hair_color":
        return hair_scene(directory)
    if name == "camera_film":
        scene = graph_smoke()
        scene["name"] = "Thin lens, filmic, thin glass and random walk skin"
        scene["camera"].update(
            aperture_radius=8,
            focus_distance=800,
            aperture_blades=5,
            aperture_rotation=0.3,
            aperture_ratio=1.4,
        )
        scene["film"].update(
            view_transform="filmic",
            exposure=0.4,
            gamma=1.2,
            contrast=1.3,
            persistence=0.91,
        )
        for material, surface in [
            (
                "tall",
                {
                    "base_color": [0.8, 0.95, 1],
                    "transmission": 0.85,
                    "thin_walled": 1,
                    "roughness": 0.12,
                    "ior": 1.45,
                },
            ),
            (
                "short",
                {
                    "base_color": [0.8, 0.4, 0.2],
                    "subsurface": 0.8,
                    "subsurface_method": 2,
                    "subsurface_scale": 30,
                    "subsurface_radius": [1, 0.4, 0.2],
                    "roughness": 0.4,
                },
            ),
        ]:
            scene["materials"][material] = {
                "type": "shader_graph",
                "graph": {"nodes": {}, "surface": surface},
            }
        return scene
    if name == "many_lights":
        scene = read_scene(ROOT / "assets/scenes/point_light/scene.json")
        scene["name"] = "70 point lights, two scan groups and mirrored instances"
        scene["entities"][0]["transform"] = {
            "scale": [-0.8, 1.2, 0.6],
            "rotation_degrees": [0, 30, 0],
        }
        scene["entities"] = [e for e in scene["entities"] if e["type"] != "point_light"]
        for i in range(70):
            angle = i * 2 * math.pi / 70
            scene["entities"].append(
                {
                    "type": "point_light",
                    "position": [3 * math.cos(angle), 3, 3 * math.sin(angle)],
                    "color": [
                        1,
                        0.5 + 0.3 * math.cos(angle),
                        0.5 + 0.3 * math.sin(angle),
                    ],
                    "strength": 3,
                    "sampling_weight": 1 + (i % 3),
                    "radius": 0 if i % 3 == 0 else 0.15,
                    "soft_falloff": i % 3 == 1,
                }
            )
        return scene
    return read_scene(ROOT / "assets/scenes" / name / "scene.json")


def read_pfm(path):
    with path.open("rb") as f:
        assert f.readline().strip() == b"PF"
        w, h = map(int, f.readline().split())
        scale = float(f.readline())
        a = np.frombuffer(f.read(), dtype="<f4" if scale < 0 else ">f4")
        a = a.reshape(h, w, 3)[::-1].astype(np.float64)
        if not np.isfinite(a).all():
            raise ValueError(f"nonfinite radiance in {path}")
        return a


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["cpu", "cuda", "vulkan", "vulkan_fallback", "vulkan_query"],
        default=["cpu", "cuda", "vulkan"],
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=list(DEMOS)
        + ["graph_smoke", "camera_film", "hair_color", "many_lights"],
    )
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument(
        "--spp", type=int, default=16, help="samples per frame (not total samples)"
    )
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--bounces", type=int, default=12)
    parser.add_argument("--max-rmse", type=float, default=0.03)
    parser.add_argument("--max-native-rmse", type=float, default=0.01)
    parser.add_argument(
        "--timeout", type=float, default=900, help="per-process time limit in seconds"
    )
    parser.add_argument(
        "--repeat",
        action="store_true",
        help="require byte-identical first-scene repeat per backend",
    )
    parser.add_argument(
        "--no-gpu-cpu",
        action="store_true",
        help="hide CUDA and Vulkan devices during CPU execution",
    )
    args = parser.parse_args()
    if min(args.size, args.spp, args.frames, args.bounces, args.timeout) <= 0:
        parser.error("render counts and timeout must be positive")
    args.output = args.output.resolve()
    args.cli = args.cli.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = []
    metrics = []
    failures = []
    config = vars(args).copy()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    (args.output / "configuration.json").write_text(json.dumps(config, indent=2))
    for scene_index, name in enumerate(args.scenes):
        folder = args.output / name
        folder.mkdir(exist_ok=True)
        scene = fixture(name, folder)
        scene["film"].update(width=args.size, height=args.size)
        scene["renderer"].update(
            samples_per_dispatch=args.spp, max_bounces=args.bounces
        )
        source = folder / "scene.json"
        source.write_text(json.dumps(scene, indent=2))
        images = {}
        linear = {}
        for backend in args.backends:
            api = "vulkan" if backend.startswith("vulkan") else backend
            pipeline = {"vulkan": "ray_tracing", "vulkan_query": "ray_query"}.get(
                backend, "rt_fallback"
            )
            command = [
                str(args.cli),
                str(source),
                "--backend",
                api,
                "--pipeline",
                pipeline,
                "--frames",
                str(args.frames),
                "-o",
                str(folder / f"{backend}.png"),
                "--linear-output",
                str(folder / f"{backend}.pfm"),
                "--profile",
                str(folder / f"{backend}.csv"),
                "--profile-cpu-only",
            ]
            if backend == "vulkan":
                command.append("--require-hardware-rt")
            env = os.environ.copy()
            if backend == "cpu" and args.no_gpu_cpu:
                env.update(
                    CUDA_VISIBLE_DEVICES="",
                    VK_DRIVER_FILES="/nonexistent",
                    VK_ICD_FILENAMES="/nonexistent",
                )

            def run(cmd, label):
                start = time.monotonic()
                timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
                with (folder / f"{label}.log").open("w") as log:
                    try:
                        code = subprocess.run(
                            cmd,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            env=env,
                            cwd=ROOT,
                            timeout=args.timeout,
                        ).returncode
                    except subprocess.TimeoutExpired:
                        log.write(f"\nRender timed out after {args.timeout}s\n")
                        code = 124
                record = {
                    "scene": name,
                    "backend": label,
                    "command": cmd,
                    "started": timestamp,
                    "seconds": time.monotonic() - start,
                    "returncode": code,
                }
                runs.append(record)
                with (args.output / "commands.jsonl").open("a") as f:
                    f.write(json.dumps(record) + "\n")
                print(
                    f'{name}/{label}: exit={code} wall={record["seconds"]:.2f}s',
                    flush=True,
                )
                return code

            if run(command, backend):
                failures.append(f"{name}/{backend}: execution failed")
                continue
            try:
                image = (
                    np.array(
                        Image.open(folder / f"{backend}.png").convert("RGB"),
                        dtype=np.float64,
                    )
                    / 255
                )
                radiance = read_pfm(folder / f"{backend}.pfm")
                if (
                    image.shape != (args.size, args.size, 3)
                    or radiance.shape != image.shape
                ):
                    raise ValueError(f"{name}/{backend}: wrong image extent")
                images[backend] = image
                linear[backend] = radiance
                if float(np.mean(linear[backend])) < 1e-5:
                    failures.append(f"{name}/{backend}: black image")
            except Exception as e:
                failures.append(str(e))
                continue
            if args.repeat and scene_index == 0:
                cmd = command.copy()
                cmd[cmd.index("-o") + 1] = str(folder / f"{backend}-repeat.png")
                cmd[cmd.index("--linear-output") + 1] = str(
                    folder / f"{backend}-repeat.pfm"
                )
                cmd[cmd.index("--profile") + 1] = str(folder / f"{backend}-repeat.csv")
                if (
                    run(cmd, backend + "-repeat")
                    or (folder / f"{backend}.png").read_bytes()
                    != (folder / f"{backend}-repeat.png").read_bytes()
                    or (folder / f"{backend}.pfm").read_bytes()
                    != (folder / f"{backend}-repeat.pfm").read_bytes()
                ):
                    failures.append(f"{name}/{backend}: deterministic repeat differs")
        for a, b in itertools.combinations(images, 2):
            d = images[a] - images[b]
            hdr = linear[a] - linear[b]
            rmse = float(np.sqrt(np.mean(d * d)))
            record = {
                "scene": name,
                "a": a,
                "b": b,
                "png_rmse": rmse,
                "png_mae": float(np.abs(d).mean()),
                "png_max": float(np.abs(d).max()),
                "linear_rmse": float(np.sqrt(np.mean(hdr * hdr))),
                "linear_mae": float(np.abs(hdr).mean()),
                "linear_mean_a": float(linear[a].mean()),
                "linear_mean_b": float(linear[b].mean()),
                "finite": True,
            }
            metrics.append(record)
            Image.fromarray(np.uint8(np.clip(np.abs(d) * 8, 0, 1) * 255)).save(
                folder / f"diff-{a}-{b}-x8.png"
            )
            limit = args.max_native_rmse if {a, b} == {"cpu", "cuda"} else args.max_rmse
            if rmse > limit:
                failures.append(f"{name}/{a}/{b}: RMSE {rmse:.6f} > {limit}")
            print(
                f'  {a}/{b}: PNG RMSE {rmse:.6f}, linear RMSE {record["linear_rmse"]:.6f}',
                flush=True,
            )
        (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))
        (args.output / "status.json").write_text(
            json.dumps({"runs": len(runs), "failures": failures}, indent=2)
        )
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"PASS: {len(runs)} runs, {len(metrics)} image comparisons")


if __name__ == "__main__":
    main()
