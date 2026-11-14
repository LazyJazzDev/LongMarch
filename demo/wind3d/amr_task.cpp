#include "amr_task.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>

#include "amr_gpu_flow.h"
#include "wall_model.h"

namespace wind3d {
int RunAdaptiveTask(int grid,
                    int steps,
                    int warmup,
                    const SolverOptions &options,
                    const AdaptiveTaskOptions &adaptive,
                    const char *csv) {
  if (options.devices != 1 || !options.corrected_transport || options.export_field || adaptive.levels < 0 ||
      adaptive.levels > 6 || adaptive.block_cells < 2 || adaptive.block_cells > 64 || adaptive.block_cells % 2 ||
      grid < 64 || grid > max_grid || grid % 16 || !std::isfinite(options.max_dt) || options.max_dt < 0 ||
      !(options.initial_controls.speed > 0)) {
    std::fputs("Adaptive batch requires one GPU, corrected transport, --no-export and valid grid/options.\n", stderr);
    return 1;
  }
  const Controls controls = options.initial_controls;
  for (float value :
       {controls.speed, controls.angle, controls.flap, controls.aileron, controls.elevator, controls.rudder})
    if (!std::isfinite(value))
      return 1;
  int divisor = adaptive.block_cells * (1 << adaptive.levels);
  if (grid % divisor || (grid / 2) % divisor || (grid * 3 / 4) % divisor) {
    std::fputs("Grid dimensions must be divisible by block-cells * 2^levels.\n", stderr);
    return 1;
  }
  auto setup_start = std::chrono::steady_clock::now();
  std::array<int, 3> roots{grid / divisor, (grid / 2) / divisor, (grid * 3 / 4) / divisor};
  std::array<double, 3> origin{-6, -4, -6};
  double root_width = 16.0 / roots[0];
  float h = 16.0f / grid;
  amr::RefinementOptions refinement;
  refinement.max_level = adaptive.levels;
  refinement.capacity = adaptive.capacity;
  refinement.reference_speed = controls.speed;
  refinement.protect_padding = 2 * h;
  // AircraftDistance's conservative body-space broad phase includes control
  // sweeps. Rotate all corners back into world space, then add stencil padding.
  amr::ProtectedBox box;
  box.lower.fill(std::numeric_limits<double>::infinity());
  box.upper.fill(-std::numeric_limits<double>::infinity());
  for (float x : {-3.1f, 3.4f})
    for (float y : {-.85f, 1.4f})
      for (float z : {-3.4f, 3.4f}) {
        auto p = RotateZ({x, y, z}, -controls.angle);
        double coordinate[3]{p.x, p.y, p.z};
        for (int a = 0; a < 3; ++a) {
          box.lower[a] = std::min(box.lower[a], coordinate[a]);
          box.upper[a] = std::max(box.upper[a], coordinate[a]);
        }
      }
  refinement.protected_regions.push_back(box);
  amr::Topology tree;
  bool built = adaptive.uniform
                   ? tree.Regrid(roots, adaptive.levels, adaptive.capacity, [](amr::Block) { return true; })
                   : amr::ProtectedTopology(roots, root_width, origin, refinement, tree);
  if (!built) {
    std::fputs("Cannot build requested hierarchy within the block capacity.\n", stderr);
    return 2;
  }
  amr::GpuFlow flow;
  if (!flow.Initialize(
          0, tree, adaptive.block_cells, root_width, origin,
          [controls](std::array<double, 3> p) {
            return double(AircraftDistance({float(p[0]), float(p[1]), float(p[2])}, controls));
          },
          controls.speed)) {
    std::fprintf(stderr, "Adaptive initialization failed: %s\n", flow.Error().c_str());
    return 2;
  }
  std::ofstream file;
  if (csv) {
    file.open(csv);
    if (!file)
      return 2;
    file << std::setprecision(17) << "step,time,dt,Fx,Fy,Fz,Mx,My,Mz,seconds,cells,generation\n";
  }
  auto info = flow.Info();
  std::printf("SETUP backend=%s finest_h=%.9g cells=%zu blocks=%zu seconds=%.9g\n",
              adaptive.uniform ? "amr-uniform" : "adaptive", h, info.cells, info.blocks,
              std::chrono::duration<double>(std::chrono::steady_clock::now() - setup_start).count());
  amr::FlowParameters physics;
  physics.cubic = options.cubic;
  physics.viscosity = options.viscosity;
  physics.wall_viscosity = options.wall_viscosity;
  physics.sponge_length = options.sponge;
  double maximum = controls.speed, total = 0, total_physical = 0;
  for (int step = 1; step <= steps; ++step) {
    auto start = std::chrono::steady_clock::now();
    float bound = std::max(float(maximum), controls.speed);
    float diffusion = std::max(options.viscosity, WallEffectiveViscosity(options.wall_viscosity, 2 * bound, .5f * h));
    float dt = std::min(options.cfl * h / bound, .08f * h * h / diffusion);
    if (options.max_dt > 0)
      dt = std::min(dt, options.max_dt);
    auto result = flow.Advance(dt, controls.speed, physics);
    if (!result.advanced || !flow.MaximumSpeed(maximum)) {
      std::fprintf(stderr, "Adaptive step %d failed: %s\n", step, flow.Error().c_str());
      return 2;
    }
    info = flow.Info();
    double adapt_seconds = 0;
    if (!adaptive.uniform && adaptive.adapt_every > 0 && step % adaptive.adapt_every == 0 && step < steps) {
      auto adapt_start = std::chrono::steady_clock::now();
      amr::RefinementReport report;
      if (!flow.Adapt(refinement, report)) {
        std::fprintf(stderr, "Adaptation failed: %s\n", flow.Error().c_str());
        return 2;
      }
      adapt_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - adapt_start).count();
      // Projection on the new hierarchy can change the CFL speed bound.
      if (report.changed && !flow.MaximumSpeed(maximum))
        return 2;
      std::printf("REGRID step=%d changed=%d blocks=%zu->%zu indicator=%.9g seconds=%.9g\n", step, int(report.changed),
                  report.old_blocks, report.new_blocks, report.max_indicator, adapt_seconds);
    }
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (step > warmup) {
      total += seconds;
      total_physical += dt;
    }
    const auto &f = result.total_loads;
    std::printf(
        "step=%d seconds=%.9g time=%.9g dt=%.9g cycles=%d residual=%.9g force=%.9g,%.9g,%.9g moment=%.9g,%.9g,%.9g cells=%zu generation=%llu max_speed=%.9g adapt=%.9g transport=%.9g pressure=%.9g loads=%.9g\n",
        step, seconds, info.time, dt, result.projection.iterations, result.projection.relative_residual, f[0], f[1],
        f[2], f[3], f[4], f[5], info.cells, (unsigned long long)info.generation, maximum, adapt_seconds,
        result.transport_seconds, result.pressure_seconds, result.loads_seconds);
    std::fflush(stdout);
    if (csv) {
      file << step << ',' << info.time << ',' << dt;
      for (double v : f)
        file << ',' << v;
      file << ',' << seconds << ',' << info.cells << ',' << info.generation << '\n';
      if (!file)
        return 2;
    }
  }
  if (csv) {
    file.flush();
    if (!file)
      return 2;
  }
  std::printf(
      "RESULT backend=%s grid=%d gpus=1 steps=%d warmup=%d export=0 mean_seconds=%.9g steps_per_second=%.9g physical_seconds_per_second=%.9g\n",
      adaptive.uniform ? "amr-uniform" : "adaptive", grid, steps, warmup, total / (steps - warmup),
      (steps - warmup) / total, total_physical / total);
  return 0;
}
}  // namespace wind3d
