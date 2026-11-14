#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>

#include "aircraft.h"
#include "amr_gpu_projection.h"
#include "amr_gpu_transport.h"

int main() {
  using namespace wind3d::amr;
  Topology tree;
  if (!tree.Regrid({8, 4, 6}, 1, 1000,
                   [](Block b) { return b.x >= 1 && b.x <= 4 && b.y >= 1 && b.y <= 2 && b.z >= 1 && b.z <= 4; }))
    return 1;
  Mesh mesh;
  if (!mesh.Build(tree, 4, 2, {-6, -4, -6}))
    return 2;
  const size_t nf = mesh.Faces().size(), nc = mesh.Cells().size(), ub = nf * sizeof(double), pb = nc * sizeof(double);

  struct Buffers {
    double *u{}, *p{};

    ~Buffers() {
      cudaFree(u);
      cudaFree(p);
    }
  } device;

  if (cudaMalloc(&device.u, ub) != cudaSuccess || cudaMalloc(&device.p, pb) != cudaSuccess)
    return 3;
  for (int scenario = 0; scenario < 3; ++scenario) {
    bool embedded = scenario == 2;
    double viscosity = scenario == 1 ? .01 : 1.5e-5;
    double sponge = scenario == 1 ? .7 : 2;
    if (embedded) {
      wind3d::Controls controls;
      controls.angle = 4;
      if (!mesh.Embed([&](std::array<double, 3> x) {
            return double(wind3d::AircraftDistance({float(x[0]), float(x[1]), float(x[2])}, controls));
          }))
        return 4;
    }
    GpuTransport gpu;
    GpuProjection projection;
    FaceGradient gradient;
    if (!gradient.Build(mesh) || !projection.Build(0, mesh, gradient) || !gpu.Build(0, mesh, tree, 2, {-6, -4, -6}))
      return 5;
    std::vector<double> u(nf), p, actual(nf), actual_p(nc);
    for (size_t f = 0; f < nf; ++f)
      u[f] = mesh.Faces()[f].axis == 0 ? 1 + (scenario == 1 ? .1 * std::sin(mesh.Faces()[f].centre[1]) : 0) : 0;
    if (cudaMemcpy(device.u, u.data(), ub, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemset(device.p, 0, pb) != cudaSuccess)
      return 6;
    for (int step = 0; step < (embedded ? 3 : 1); ++step) {
      auto local_u = step ? actual : u;
      std::vector<double> local_p;
      auto local = AdvanceReference(mesh, tree, 2, {-6, -4, -6}, .01, 1, local_u, local_p, true, 1.225, 9.2e-7,
                                    viscosity, sponge);
      if (!local.advanced)
        return 14;
      auto ref = AdvanceReference(mesh, tree, 2, {-6, -4, -6}, .01, 1, u, p, true, 1.225, 9.2e-7, viscosity, sponge);
      auto result =
          gpu.AdvanceDevice(projection, device.u, nf, device.p, nc, .01, 1, true, 1.225, 9.2e-7, viscosity, sponge);
      if (!ref.advanced || !result.advanced) {
        std::cerr << "advance failed " << gpu.Error() << '\n';
        return 7;
      }
      if (cudaMemcpy(actual.data(), device.u, ub, cudaMemcpyDeviceToHost) != cudaSuccess ||
          cudaMemcpy(actual_p.data(), device.p, pb, cudaMemcpyDeviceToHost) != cudaSuccess)
        return 8;
      double velocity_error = 0, pressure_error = 0, load_error = 0;
      for (size_t f = 0; f < nf; ++f)
        velocity_error = std::max(velocity_error, std::abs(u[f] - actual[f]));
      for (size_t c = 0; c < nc; ++c)
        pressure_error = std::max(pressure_error, std::abs(p[c] - actual_p[c]));
      for (int k = 0; k < 6; ++k) {
        double pe = std::abs(result.pressure_loads[k] - ref.pressure_loads[k]);
        double fe = std::abs(result.friction_loads[k] - ref.friction_loads[k]);
        load_error = std::max({load_error, pe, fe});
        if (pe > 1e-5 + 2e-6 * std::abs(ref.pressure_loads[k]) || fe > 1e-8 + 2e-6 * std::abs(ref.friction_loads[k]))
          return 9;
        if (result.total_loads[k] != result.pressure_loads[k] + result.friction_loads[k])
          return 10;
      }
      std::cout << "GPU scenario=" << scenario << " embedded=" << embedded << " step=" << step + 1
                << " velocity error=" << velocity_error << " pressure error=" << pressure_error
                << " load error=" << load_error << '\n';
      // Separately bound accumulated differences: limiter decisions can amplify
      // tiny input perturbations even when the same-input operators agree.
      if (velocity_error > 1e-4 || pressure_error > 1e-3)
        return 15;
      double local_error = 0, local_pressure_error = 0;
      for (size_t f = 0; f < nf; ++f)
        local_error = std::max(local_error, std::abs(local_u[f] - actual[f]));
      for (size_t c = 0; c < nc; ++c)
        local_pressure_error = std::max(local_pressure_error, std::abs(local_p[c] - actual_p[c]));
      std::cout << "same-input local velocity error=" << local_error << " pressure error=" << local_pressure_error
                << '\n';
      if (local_error > 5e-7 || local_pressure_error > 5e-5)
        return 11;
    }
    auto saved = actual, saved_p = actual_p;
    if (gpu.AdvanceDevice(projection, device.u, nf, device.p, nc, -1, 1).advanced ||
        gpu.AdvanceDevice(projection, device.u, nf, device.p, nc, .01, 1, true, 1.225, 9.2e-7, 100).advanced)
      return 12;
    if (cudaMemcpy(actual.data(), device.u, ub, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(actual_p.data(), device.p, pb, cudaMemcpyDeviceToHost) != cudaSuccess || actual != saved ||
        actual_p != saved_p)
      return 13;
  }
  std::cout
      << "PASS resident GPU timestep, aircraft pressure/friction loads and invalid-step rollback (not steady-load validation)\n";
}
