#pragma once
#include "simulation.h"

namespace wind3d {
struct AdaptiveTaskOptions {
  int levels{1}, block_cells{4}, adapt_every{100};
  size_t capacity{100000};
  bool uniform{}, compatible_pressure{};
};

int RunAdaptiveTask(int grid,
                    int steps,
                    int warmup,
                    const SolverOptions &options,
                    const AdaptiveTaskOptions &adaptive,
                    const char *csv);
}  // namespace wind3d
