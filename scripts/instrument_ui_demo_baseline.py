"""Instrument an isolated e233d265 baseline for benchmark_ui_demos.py.

Only measurement, presentation suppression and a reproducible spawn seed change;
the graphics renderer and game logic remain the baseline implementation.
"""

import argparse
from pathlib import Path
import shutil


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected one original occurrence of {old!r}; use a fresh baseline")
    return source.replace(old, new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_root", type=Path)
    args = parser.parse_args()
    edits = {}
    for demo in ("2048", "gol"):
        path = args.baseline_root / "demo" / demo / "application/application.cpp"
        source = path.read_text(encoding="utf-8-sig")
        for old, new in (
            ('#include <algorithm>', '#include <algorithm>\n#include "../../frame_benchmark.h"'),
            ('  int frames = 0;', '  int frames = 0;\n  demo::FrameBenchmark benchmark;'),
            ('    grassland::graphics::Window::PollEvents();',
             '    if (frames == 120) core_->WaitGPU();\n'
             '    const bool measured = benchmark.BeginFrame(frames);\n'
             '    grassland::graphics::Window::PollEvents();'),
            ('    frames++;', '    benchmark.EndFrame(measured);\n    frames++;'),
            ('  if (!screenshot_path_.empty()) {',
             '  core_->WaitGPU();\n  benchmark.Finish();\n  if (!screenshot_path_.empty()) {'),
            ('  context->CmdPresent(window_.get(), present_image_.get());',
             '  if (demo::FrameBenchmark::PresentFrames())\n'
             '    context->CmdPresent(window_.get(), present_image_.get());'),
        ):
            source = replace_once(source, old, new)
        edits[path] = source
    path = args.baseline_root / "demo/2048/2048.cpp"
    source = path.read_text(encoding="utf-8-sig")
    source = replace_once(source, '#include <algorithm>', '#include <algorithm>\n#include <cstdlib>')
    edits[path] = replace_once(
        source, '  random_device_ = std::mt19937(int(std::time(nullptr)));',
        '  const char *validation_seed = std::getenv("LONGMARCH_DEMO_SEED");\n'
        '  random_device_ = std::mt19937(validation_seed ? uint32_t(std::strtoul(validation_seed, nullptr, 10))\n'
        '                                              : uint32_t(std::time(nullptr)));')
    for path, source in edits.items():
        path.write_text(source, encoding="utf-8-sig")
    shutil.copyfile(Path(__file__).resolve().parent.parent / "demo/frame_benchmark.h",
                    args.baseline_root / "demo/frame_benchmark.h")


if __name__ == "__main__":
    main()
