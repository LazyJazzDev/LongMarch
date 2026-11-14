#include "amr_gpu_flow.h"

#include <algorithm>
#include <cmath>
#include <iostream>

bool Same(const wind3d::amr::FlowSnapshot &a, const wind3d::amr::FlowSnapshot &b) {
  return a.velocity == b.velocity && a.pressure == b.pressure && a.topology.Leaves() == b.topology.Leaves() &&
         a.info.steps == b.info.steps && a.info.time == b.info.time && a.info.generation == b.info.generation &&
         a.info.pressure_valid == b.info.pressure_valid;
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

int Run() {
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
    if (!flow.Initialize(0, body ? mixed : coarse, 4, 1, {-1, -1, -1}, distance, 1) || !flow.Readback(state))
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
    if (flow.Initialize(0, coarse, 3, 1, {-1, -1, -1}, distance, 1) || !flow.Readback(state) || !Same(before, state))
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
    if (!flow.Advance(.01, 1).advanced)
      return 22;
  }
  std::cout << "PASS transactional runtime GPU regrids, body preservation, clock continuity and pressure validity\n";
  return 0;
}

int main() {
  int code = Run();
  if (code)
    std::cerr << "GPU flow lifecycle failure code=" << code << '\n';
  return code;
}
