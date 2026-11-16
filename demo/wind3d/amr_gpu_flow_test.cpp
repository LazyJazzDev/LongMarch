#include "amr_gpu_flow.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>

bool Same(const wind3d::amr::FlowSnapshot &a, const wind3d::amr::FlowSnapshot &b) {
  return a.compatible_pressure == b.compatible_pressure && a.velocity == b.velocity && a.pressure == b.pressure &&
         a.topology.Leaves() == b.topology.Leaves() && a.info.steps == b.info.steps && a.info.time == b.info.time &&
         a.info.generation == b.info.generation && a.info.pressure_valid == b.info.pressure_valid;
}

double Divergence(const wind3d::amr::FlowSnapshot &s) {
  double error = 0;
  for (size_t c = 0; c < s.mesh.Cells().size(); ++c) {
    double flux = 0;
    for (int k = s.mesh.Offsets()[c]; k < s.mesh.Offsets()[c + 1]; ++k) {
      int id = s.mesh.Incidence()[k], i = std::abs(id) - 1;
      auto f = s.mesh.Faces()[i];
      flux += (id > 0 ? 1 : -1) * f.area * f.aperture * s.velocity[i];
    }
    error = std::max(error, std::abs(flux) / std::pow(s.mesh.Cells()[c].width, 3));
  }
  return error;
}

int Run(bool compatible) {
  using namespace wind3d::amr;
  Topology coarse, fine, mixed, wrong;
  if (!coarse.Regrid({2, 2, 2}, 0, 8, [](Block) { return false; }) ||
      !fine.Regrid({2, 2, 2}, 1, 64, [](Block) { return true; }) ||
      !mixed.Regrid({2, 2, 2}, 1, 64, [](Block b) { return b.x == 0; }) ||
      !wrong.Regrid({3, 2, 2}, 0, 12, [](Block) { return false; }))
    return 1;
  for (bool body : {false, true}) {
    GpuFlow flow;
    FlowSnapshot state, before;
    GpuFlow::Distance distance;
    if (body)
      distance = [](std::array<double, 3> p) {
        return std::max({std::abs(p[0] + .5), std::abs(p[1]), std::abs(p[2])}) - .21;
      };
    if (!flow.Initialize(0, body ? mixed : coarse, 4, 1, {-1, -1, -1}, distance, 1, compatible) ||
        !flow.Readback(state))
      return 2;
    double maximum;
    if (!flow.MaximumSpeed(maximum) || std::abs(maximum - 1) > 1e-12)
      return 23;
    if (state.info.pressure_valid || state.info.steps || state.info.time)
      return 3;
    if (!flow.Advance(.01, 1).advanced || !flow.Readback(before))
      return 4;
    if (!flow.Regrid(fine) || !flow.Readback(state)) {
      std::cerr << flow.Error() << '\n';
      return 5;
    }
    if (state.info.steps != before.info.steps || state.info.time != before.info.time ||
        state.info.generation != before.info.generation + 1 || state.info.pressure_valid)
      return 6;
    for (double p : state.pressure)
      if (p != 0)
        return 7;
    double div = Divergence(state);
    std::cout << "regrid body=" << body << " max divergence=" << div << '\n';
    if (div > 1e-6)
      return 8;
    if (!body)
      for (size_t f = 0; f < state.velocity.size(); ++f)
        if (std::abs(state.velocity[f] - (state.mesh.Faces()[f].axis == 0 ? 1 : 0)) > 1e-12)
          return 9;
    before = state;
    if (!flow.Regrid(fine) || !flow.Readback(state) || !Same(before, state))
      return 10;
    if (!flow.Advance(.01, 1).advanced || !flow.Regrid(mixed) || !flow.Readback(state)) {
      std::cerr << "post-refine advance/coarsen: " << flow.Error() << '\n';
      return 11;
    }
    if (state.info.steps != 2 || state.info.time != .02 || state.info.generation != 2 || state.info.pressure_valid ||
        Divergence(state) > 1e-6)
      return 12;
    before = state;
    if (flow.Regrid(wrong) || flow.Error().empty() || !flow.Readback(state) || !Same(before, state))
      return 13;
    if (flow.Initialize(0, coarse, 3, 1, {-1, -1, -1}, distance, 1, compatible) || !flow.Readback(state) ||
        !Same(before, state))
      return 14;
    if (body && (flow.Regrid(coarse) || !flow.Readback(state) || !Same(before, state)))
      return 15;
    if (flow.Advance(-1, 1).advanced || !flow.Readback(state) || !Same(before, state))
      return 16;
    if (!flow.Advance(.01, 1).advanced || !flow.Readback(state) || !state.info.pressure_valid || state.info.steps != 3)
      return 17;
    before = state;
    RefinementOptions options;
    options.max_level = 1;
    options.capacity = 64;
    options.buffer_blocks = 0;
    if (body)
      options.protected_regions.push_back({{-.8, -.3, -.3}, {-.2, .3, .3}});
    RefinementReport report;
    if (!flow.Adapt(options, report) || !flow.Readback(state)) {
      std::cerr << flow.Error() << '\n';
      return 18;
    }
    if (state.info.steps != before.info.steps || state.info.time != before.info.time ||
        report.new_blocks != state.info.blocks || report.old_blocks != before.info.blocks ||
        state.info.generation != before.info.generation + size_t(report.changed))
      return 19;
    if (report.changed && state.info.pressure_valid)
      return 20;
    before = state;
    auto saved_report = report;
    options.capacity = 1;
    if (flow.Adapt(options, report) || !flow.Readback(state) || !Same(before, state) ||
        report.new_blocks != saved_report.new_blocks)
      return 21;
    FlowParameters parameters;
    parameters.pressure_tolerance = 0;
    if (flow.Advance(.01, 1, parameters).advanced || !flow.Readback(state) || !Same(before, state))
      return 24;
    parameters.pressure_tolerance = 1e-11;
    auto tight = flow.Advance(.01, 1, parameters);
    if (!tight.advanced || tight.projection.relative_residual > parameters.pressure_tolerance)
      return 22;
  }
  std::cout << "PASS transactional runtime GPU regrids, body preservation, clock continuity and pressure validity\n";
  return 0;
}

int RestoreContinuation() {
  using namespace wind3d::amr;
  Topology fine, coarse;
  if (!fine.Regrid({2, 2, 2}, 1, 64, [](Block) { return true; }) ||
      !coarse.Regrid({2, 2, 2}, 1, 64, [](Block) { return false; }))
    return 30;
  namespace fs = std::filesystem;
  std::error_code ec;
  auto directory =
      fs::temp_directory_path(ec) /
      ("wind3d-checkpoint-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  if (ec || !fs::create_directory(directory, ec))
    return 36;

  struct Cleanup {
    fs::path path;

    ~Cleanup() {
      std::error_code error;
      fs::remove_all(path, error);
    }
  } cleanup{directory};

  auto checkpoint = directory / "state.bin", damaged = directory / "damaged.bin";
  for (bool compatible : {false, true})
    for (bool body : {false, true}) {
      GpuFlow original, resumed;
      GpuFlow::Distance distance;
      if (body)
        distance = [](std::array<double, 3> p) {
          return std::max({std::abs(p[0]), std::abs(p[1]), std::abs(p[2])}) - .21;
        };
      if (!original.Initialize(0, fine, 4, 1, {-1, -1, -1}, distance, 1, compatible) ||
          !resumed.Initialize(0, body ? fine : coarse, 4, 1, {-1, -1, -1}, distance, 1, compatible) ||
          !original.Advance(.01, 1).advanced)
        return 31;
      FlowSnapshot saved, actual, expected;
      if (!original.Readback(saved) || !resumed.Restore(saved) || !resumed.Readback(actual) || !Same(saved, actual))
        return 32;
      if (!original.SaveCheckpoint(checkpoint.string()) || !resumed.LoadCheckpoint(checkpoint.string()) ||
          !resumed.Readback(actual) || !Same(saved, actual))
        return 37;
      for (int step = 0; step < 3; ++step) {
        double dt = .005 + .001 * step, speed = 1. + .1 * step;
        auto a = original.Advance(dt, speed), b = resumed.Advance(dt, speed);
        if (!a.advanced || !b.advanced || a.total_loads != b.total_loads || !original.Readback(expected) ||
            !resumed.Readback(actual) || !Same(expected, actual))
          return 33;
      }
      // Existing checkpoint replacement must contain the newer clock and fields.
      if (!original.SaveCheckpoint(checkpoint.string()) || !resumed.LoadCheckpoint(checkpoint.string()) ||
          !resumed.Readback(actual) || !Same(expected, actual))
        return 38;
      if (original.SaveCheckpoint((directory / "missing" / "state.bin").string()) ||
          !resumed.LoadCheckpoint(checkpoint.string()))
        return 39;
      for (int corruption = 0; corruption < 3; ++corruption) {
        fs::copy_file(checkpoint, damaged, fs::copy_options::overwrite_existing, ec);
        if (ec)
          return 40;
        auto size = fs::file_size(damaged, ec);
        if (ec)
          return 40;
        if (corruption == 0) {
          fs::resize_file(damaged, size - 1, ec);
          if (ec)
            return 40;
        } else if (corruption == 1) {
          std::ofstream file(damaged, std::ios::binary | std::ios::app);
          file.put('x');
          if (!file)
            return 40;
        } else {
          std::fstream file(damaged, std::ios::binary | std::ios::in | std::ios::out);
          file.seekg(std::streamoff(size - 9));
          char byte;
          if (!file.get(byte))
            return 40;
          file.seekp(std::streamoff(size - 9));
          file.put(char(static_cast<unsigned char>(byte) ^ 1));
          if (!file)
            return 40;
        }
        if (resumed.LoadCheckpoint(damaged.string()) || !resumed.Readback(actual) || !Same(expected, actual))
          return 41;
      }
      for (int invalid = 0; invalid < 5; ++invalid) {
        auto bad = saved;
        if (invalid == 0)
          bad.info.time = -1;
        else if (invalid == 1)
          bad.velocity[0] = std::numeric_limits<double>::quiet_NaN();
        else if (invalid == 2)
          bad.compatible_pressure = !compatible;
        else if (invalid == 3)
          bad.pressure.pop_back();
        else if (!bad.mesh.Build(fine, 4, 2, {-1, -1, -1}))
          return 34;
        if (resumed.Restore(bad) || resumed.Error().empty() || !resumed.Readback(actual) || !Same(expected, actual))
          return 35;
      }
    }
  std::cout << "PASS snapshot/file restoration, exact continuation, corrupt-file rejection and rollback\n";
  return 0;
}

int main() {
  int code = Run(false);
  if (!code)
    code = Run(true);
  if (!code)
    code = RestoreContinuation();
  if (code)
    std::cerr << "GPU flow lifecycle failure code=" << code << '\n';
  return code;
}
