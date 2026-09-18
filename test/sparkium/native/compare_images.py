#!/usr/bin/env python3
"""Compares two renders of the same scene and reports pixel difference stats.

Usage: compare_images.py reference.png candidate.png [--diff diff.png] [--json out.json]

The renders are Monte Carlo estimates, so an exact match is only expected when
both backends run the same sample sequence with the same arithmetic. The stats
below are the ones quoted in /results/REPORT.md: mean/max absolute difference
on 8-bit channels, PSNR, and the fraction of pixels differing by more than a
few levels.
"""
import argparse
import json
import sys

import numpy as np
from PIL import Image


def load(path):
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    return image


def block_stats(reference, candidate, block):
    """Averages both images over block x block tiles before differencing."""
    height, width = reference.shape[:2]
    height -= height % block
    width -= width % block
    if height == 0 or width == 0:
        return {"mean_abs": None, "max_abs": None}
    shape = (height // block, block, width // block, block, reference.shape[2])
    pooled_reference = reference[:height, :width].reshape(shape).mean(axis=(1, 3))
    pooled_candidate = candidate[:height, :width].reshape(shape).mean(axis=(1, 3))
    difference = np.abs(pooled_reference - pooled_candidate)
    return {"mean_abs": float(np.mean(difference)), "max_abs": float(np.max(difference))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--diff")
    parser.add_argument("--json")
    parser.add_argument("--amplify", type=float, default=8.0)
    parser.add_argument("--block", type=int, default=16)
    parser.add_argument("--max-mean", type=float)
    parser.add_argument("--max-block-mean", type=float)
    parser.add_argument("--max-psnr-floor", type=float)
    arguments = parser.parse_args()

    reference = load(arguments.reference)
    candidate = load(arguments.candidate)
    if reference.shape != candidate.shape:
        print(f"shape mismatch: {reference.shape} vs {candidate.shape}", file=sys.stderr)
        return 2

    difference = np.abs(reference - candidate)
    mse = float(np.mean((reference - candidate) ** 2))
    psnr = float("inf") if mse == 0.0 else 10.0 * np.log10(255.0 * 255.0 / mse)
    block = block_stats(reference, candidate, arguments.block)
    stats = {
        "reference": arguments.reference,
        "candidate": arguments.candidate,
        "width": int(reference.shape[1]),
        "height": int(reference.shape[0]),
        "mean_abs": float(np.mean(difference)),
        "max_abs": float(np.max(difference)),
        "rmse": float(np.sqrt(mse)),
        "psnr_db": psnr,
        "identical": bool(np.array_equal(reference, candidate)),
        "fraction_gt_1": float(np.mean(np.any(difference > 1.0, axis=2))),
        "fraction_gt_4": float(np.mean(np.any(difference > 4.0, axis=2))),
        "fraction_gt_16": float(np.mean(np.any(difference > 16.0, axis=2))),
        "mean_reference": float(np.mean(reference)),
        "mean_candidate": float(np.mean(candidate)),
        "mean_signed": float(np.mean(candidate - reference)),
        # Block-averaged statistics. Independent Monte Carlo noise shrinks like
        # 1/block when averaged over a block x block tile, so these separate a
        # systematic difference from a different sample sequence.
        "block_size": arguments.block,
        "block_mean_abs": block["mean_abs"],
        "block_max_abs": block["max_abs"],
        # A difference of at most one 8-bit level is the quantization step of
        # the develop pass itself and cannot be attributed to the renderer.
        "fraction_le_1": float(np.mean(np.all(difference <= 1.0, axis=2))),
    }
    print(json.dumps(stats, indent=2))
    if arguments.json:
        with open(arguments.json, "w") as stream:
            json.dump(stats, stream, indent=2)
            stream.write("\n")
    if arguments.diff:
        amplified = np.clip(difference * arguments.amplify, 0.0, 255.0).astype(np.uint8)
        Image.fromarray(amplified).save(arguments.diff)

    status = 0
    if arguments.max_mean is not None and stats["mean_abs"] > arguments.max_mean:
        print(f"FAIL: mean_abs {stats['mean_abs']:.4f} > {arguments.max_mean}", file=sys.stderr)
        status = 1
    if arguments.max_block_mean is not None and stats["block_mean_abs"] > arguments.max_block_mean:
        print(f"FAIL: block_mean_abs {stats['block_mean_abs']:.4f} > {arguments.max_block_mean}", file=sys.stderr)
        status = 1
    if arguments.max_psnr_floor is not None and stats["psnr_db"] < arguments.max_psnr_floor:
        print(f"FAIL: psnr {stats['psnr_db']:.2f} < {arguments.max_psnr_floor}", file=sys.stderr)
        status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
