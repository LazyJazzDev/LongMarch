#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <thread>

#include "amr_task.h"
#include "validation.h"

int main(int argc, char **argv) {
  int grid = 192, steps = 80, warmup = 10;
  auto options = wind3d::OptimizedSolverOptions();
  bool validate = false;
  const char *dump = nullptr, *csv = nullptr;
  bool adaptive_mode = false;
  wind3d::AdaptiveTaskOptions adaptive;
  for (int i = 1; i < argc; ++i) {
    const char *arg = argv[i];
    if (!std::strcmp(arg, "--adaptive") || !std::strcmp(arg, "--amr-uniform")) {
      adaptive_mode = true;
      adaptive.uniform = !std::strcmp(arg, "--amr-uniform");
      continue;
    }
    if (!std::strcmp(arg, "--amr-compatible-pressure")) {
      adaptive.compatible_pressure = true;
      continue;
    }
    if (!std::strcmp(arg, "--validate")) {
      validate = true;
      continue;
    }
    if (!std::strcmp(arg, "--no-export")) {
      options.export_field = false;
      continue;
    }
    if (!std::strcmp(arg, "--cubic")) {
      options.cubic = true;
      continue;
    }
    if (!std::strcmp(arg, "--first-order")) {
      options.corrected_transport = false;
      continue;
    }
    if (!std::strcmp(arg, "--help")) {
      std::puts(
          "wind3d_task [--grid 192] [--gpus 1|2|4|8] [--steps 80] [--warmup 10]\n"
          "            [--pressure-tolerance 0.0001] [--pressure-cycles 64]\n"
          "            [--cubic] [--first-order] [--validate] [--no-export] [--dump snapshot.bin]\n"
          "            [--speed 12] [--angle 5] [--flap 0] [--aileron 0] [--elevator 0] [--rudder 0]\n"
          "            [--max-dt 0] [--csv loads.csv]\n"
          "            [--adaptive | --amr-uniform] [--levels 1] [--block-cells 4]\n"
          "            [--amr-compatible-pressure] (requires --adaptive or --amr-uniform)\n"
          "            [--adapt-every 100] [--max-blocks 100000] (adaptive requires --no-export)\n"
          "Times include transport, pressure, diagnostics and optional snapshot export; startup steps are excluded.");
      return 0;
    }
    if (++i == argc) {
      std::fprintf(stderr, "Missing value for %s\n", arg);
      return 1;
    }
    char *end = nullptr;
    if (!std::strcmp(arg, "--csv")) {
      csv = argv[i];
      continue;
    }
    if (!std::strcmp(arg, "--dump")) {
      dump = argv[i];
      continue;
    }
    float *parameter = nullptr;
    if (!std::strcmp(arg, "--max-dt"))
      parameter = &options.max_dt;
    if (!std::strcmp(arg, "--speed"))
      parameter = &options.initial_controls.speed;
    if (!std::strcmp(arg, "--angle"))
      parameter = &options.initial_controls.angle;
    if (!std::strcmp(arg, "--flap"))
      parameter = &options.initial_controls.flap;
    if (!std::strcmp(arg, "--aileron"))
      parameter = &options.initial_controls.aileron;
    if (!std::strcmp(arg, "--elevator"))
      parameter = &options.initial_controls.elevator;
    if (!std::strcmp(arg, "--rudder"))
      parameter = &options.initial_controls.rudder;
    if (parameter) {
      *parameter = std::strtof(argv[i], &end);
      if (!std::isfinite(*parameter))
        return 1;
    } else if (!std::strcmp(arg, "--pressure-tolerance")) {
      options.pressure_tolerance = std::strtof(argv[i], &end);
    } else {
      long value = std::strtol(argv[i], &end, 10);
      if (value < 0 || value > 1000000)
        return 1;
      if (!std::strcmp(arg, "--levels"))
        adaptive.levels = int(value);
      else if (!std::strcmp(arg, "--block-cells"))
        adaptive.block_cells = int(value);
      else if (!std::strcmp(arg, "--adapt-every"))
        adaptive.adapt_every = int(value);
      else if (!std::strcmp(arg, "--max-blocks"))
        adaptive.capacity = size_t(value);
      else if (!std::strcmp(arg, "--grid"))
        grid = int(value);
      else if (!std::strcmp(arg, "--gpus"))
        options.devices = int(value);
      else if (!std::strcmp(arg, "--steps"))
        steps = int(value);
      else if (!std::strcmp(arg, "--warmup"))
        warmup = int(value);
      else if (!std::strcmp(arg, "--pressure-cycles"))
        options.pressure_cycles = int(value);
      else {
        std::fprintf(stderr, "Unknown argument: %s\n", arg);
        return 1;
      }
    }
    if (!end || end == argv[i] || *end)
      return 1;
  }
  if ((validate && steps < 2) || steps <= warmup || !std::isfinite(options.pressure_tolerance))
    return 1;
  if (adaptive.compatible_pressure && !adaptive_mode) {
    std::fputs("--amr-compatible-pressure requires --adaptive or --amr-uniform.\n", stderr);
    return 1;
  }
  if (adaptive_mode) {
    if (dump || validate) {
      std::fputs("Adaptive dumps and --validate are not supported; use AMR CTests.\n", stderr);
      return 1;
    }
    return wind3d::RunAdaptiveTask(grid, steps, warmup, options, adaptive, csv);
  }
  std::ofstream csv_file;
  if (csv) {
    csv_file.open(csv);
    if (!csv_file)
      return 2;
    csv_file << std::setprecision(17) << "step,time,dt,Fx,Fy,Fz,Mx,My,Mz,seconds,cells,generation\n";
  }
  options.stop_after = steps;
  options.retain_snapshots = true;
  options.export_final = dump != nullptr;
  if (validate &&
      (!wind3d::ValidateGeometry() || !wind3d::ValidateWallDiffusion(0) || !wind3d::ValidateClosedSurface(0, true)))
    return 2;
  auto simulation = wind3d::Simulation::Create(0, grid, options);
  if (!simulation) {
    std::fputs("Cannot initialize CUDA solver; check grid, options and visible GPU count.\n", stderr);
    return 2;
  }
  uint64_t last = 0;
  double total = 0;
  std::shared_ptr<const wind3d::Snapshot> state;
  auto deadline = std::chrono::steady_clock::now() + std::chrono::hours(12);
  while (last < uint64_t(steps)) {
    auto error = simulation->Error();
    if (!error.empty()) {
      std::fprintf(stderr, "%s\n", error.c_str());
      return 2;
    }
    auto snapshot = simulation->TakeNext();
    if (snapshot && snapshot->step != last) {
      if (snapshot->step != last + 1) {
        std::fputs("Missed a snapshot; timing sample is incomplete.\n", stderr);
        return 2;
      }
      state = snapshot;
      last = state->step;
      if (!state->pressure_converged || !std::isfinite(state->divergence_after) ||
          state->divergence_after > std::max(1e-5f, 2 * options.pressure_tolerance * state->divergence_before)) {
        std::fprintf(
            stderr,
            "Projection did not converge at step %llu: cycles=%d residual=%.9g divergence_before=%.9g divergence_after=%.9g\n",
            (unsigned long long)last, state->pressure_cycles, state->pressure_residual, state->divergence_before,
            state->divergence_after);
        return 2;
      }
      if (last > uint64_t(warmup))
        total += state->step_seconds;
      std::printf(
          "step=%llu seconds=%.9g time=%.9g dt=%.9g cycles=%d residual=%.9g divergence=%.9g force=%.9g,%.9g,%.9g transport=%.9g pressure=%.9g diagnostics=%.9g\n",
          (unsigned long long)last, state->step_seconds, state->time, state->dt, state->pressure_cycles,
          state->pressure_residual, state->divergence_after, state->force.x, state->force.y, state->force.z,
          state->transport_seconds, state->pressure_seconds, state->diagnostics_seconds);
      std::fflush(stdout);
      if (csv) {
        csv_file << last << ',' << state->time << ',' << state->dt << ',' << state->force.x << ',' << state->force.y
                 << ',' << state->force.z << ',' << state->moment.x << ',' << state->moment.y << ',' << state->moment.z
                 << ',' << state->step_seconds << ',' << size_t(grid) * (grid / 2) * (grid * 3 / 4) << ','
                 << state->generation << '\n';
        if (!csv_file)
          return 2;
      }
    }
    if (std::chrono::steady_clock::now() > deadline)
      return 2;
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  if (dump) {
    std::ofstream file(dump, std::ios::binary);
    int shape[] = {state->nx, state->ny, state->nz};
    file.write(reinterpret_cast<const char *>(shape), sizeof(shape));
    file.write(reinterpret_cast<const char *>(state->cells.data()), state->cells.size() * sizeof(wind3d::Cell));
    if (!file)
      return 2;
  }
  if (validate && !wind3d::ValidateLifecycle(*simulation, state->controls))
    return 2;
  if (csv) {
    csv_file.flush();
    if (!csv_file)
      return 2;
  }
  std::printf("RESULT grid=%d gpus=%d steps=%d warmup=%d export=%d mean_seconds=%.9g steps_per_second=%.9g\n", grid,
              options.devices, steps, warmup, int(options.export_field), total / (steps - warmup),
              (steps - warmup) / total);
}
