#!/usr/bin/env python3
"""Compares two float accumulation films written by `sparkium_film_probe`.

The films are the renderer's own output, before `film2img` averaging and before
the 8-bit develop step, so the statistics here isolate the renderers from the
quantization of the output PNG.
"""
import argparse
import json
import sys

import numpy as np


def load(path):
    with open(path, "rb") as stream:
        width, height = np.frombuffer(stream.read(8), dtype=np.int32)
        pixels = np.frombuffer(stream.read(), dtype=np.float32)
    return pixels.reshape(int(height), int(width), 4)[:, :, :3].astype(np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--json")
    parser.add_argument("--block", type=int, default=16)
    arguments = parser.parse_args()

    reference = load(arguments.reference)
    candidate = load(arguments.candidate)
    if reference.shape != candidate.shape:
        print(f"shape mismatch: {reference.shape} vs {candidate.shape}", file=sys.stderr)
        return 2

    difference = candidate - reference
    mean = float(np.mean(reference))
    block = arguments.block
    height = reference.shape[0] - reference.shape[0] % block
    width = reference.shape[1] - reference.shape[1] % block
    shape = (height // block, block, width // block, block, 3)
    pooled = np.abs(
        candidate[:height, :width].reshape(shape).mean(axis=(1, 3))
        - reference[:height, :width].reshape(shape).mean(axis=(1, 3))
    )
    stats = {
        "reference": arguments.reference,
        "candidate": arguments.candidate,
        "mean_reference": mean,
        "mean_candidate": float(np.mean(candidate)),
        "mean_abs": float(np.mean(np.abs(difference))),
        "mean_signed": float(np.mean(difference)),
        "max_abs": float(np.max(np.abs(difference))),
        "identical": bool(np.array_equal(reference, candidate)),
        # Relative measures, since the film is unbounded radiance rather than
        # display levels.
        "relative_mean_abs": float(np.mean(np.abs(difference))) / mean if mean else None,
        "relative_mean_signed": float(np.mean(difference)) / mean if mean else None,
        "block_size": block,
        "relative_block_mean_abs": float(np.mean(pooled)) / mean if mean else None,
        "relative_block_max_abs": float(np.max(pooled)) / mean if mean else None,
    }
    print(json.dumps(stats, indent=2))
    if arguments.json:
        with open(arguments.json, "w") as stream:
            json.dump(stats, stream, indent=2)
            stream.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
