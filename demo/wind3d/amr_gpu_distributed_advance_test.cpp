#include "amr_gpu_distributed_advance.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

#include "amr_gpu_projection.h"
#include "amr_gpu_transport.h"

namespace {
struct Buffers {
  std::vector<std::pair<int, double *>> allocations;

  ~Buffers() {
    int old;
    cudaGetDevice(&old);
    for (auto [d, p] : allocations) {
      cudaSetDevice(d);
      cudaFree(p);
    }
    cudaSetDevice(old);
  }

  bool Allocate(int d, size_t n, double *&p) {
    if (cudaSetDevice(d) != cudaSuccess || cudaMalloc(&p, n * sizeof(double)) != cudaSuccess)
      return false;
    allocations.push_back({d, p});
    return true;
  }
};

int Run(bool mixed, bool body, double width, const std::vector<int> &devices) {
  using namespace wind3d::amr;
  Topology tree;
  Mesh mesh;
  FaceGradient unused;
  if (!tree.Regrid({2, 2, 2}, 1, 64, [mixed](Block b) { return !mixed || (b.x == 0 && b.y == 0); }) ||
      !mesh.Build(tree, 2, width, {-width, -width, -width}))
    return 1;
  if (body && !mesh.Embed([width](std::array<double, 3> p) {
        return std::max({std::abs(p[0] + .5 * width), std::abs(p[1] + .5 * width), std::abs(p[2] + .5 * width)}) -
               .22 * width;
      }))
    return 2;
  GpuDistributedAdvance gpu;
  GpuTransport single;
  GpuProjection projection;
  if (!gpu.Build(mesh, tree, devices, width, {-width, -width, -width}) ||
      !single.Build(devices[0], mesh, tree, width, {-width, -width, -width}) ||
      !projection.Build(devices[0], mesh, unused, true, true, true))
    return 3;
  auto layouts = gpu.Layouts();
  if (gpu.Build(mesh, tree, {}, width, {}) || gpu.Error().empty() || gpu.Layouts().size() != layouts.size())
    return 4;
  Buffers buffers;
  std::vector<double *> u(devices.size()), p(devices.size());
  double *single_u{}, *single_p{};
  std::vector<double> initial(mesh.Faces().size()), expected_u(initial.size()), expected_p(mesh.Cells().size());
  for (size_t i = 0; i < initial.size(); ++i) {
    auto f = mesh.Faces()[i];
    initial[i] = f.aperture > 0 ? (f.axis == 0 ? 1. : 0.) + .02 * std::sin(.17 * double(i)) : 0;
  }
  if (!buffers.Allocate(devices[0], initial.size(), single_u) ||
      !buffers.Allocate(devices[0], expected_p.size(), single_p) ||
      cudaMemcpy(single_u, initial.data(), initial.size() * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaDeviceSynchronize() != cudaSuccess)
    return 5;
  std::vector<std::vector<double>> before_u(devices.size()), before_p(devices.size());
  for (size_t r = 0; r < devices.size(); ++r) {
    auto &l = layouts[r];
    before_u[r].assign(l.faces.size() + 3, -54321.);
    before_p[r].assign(l.cells.size() + 3, -54321.);
    for (size_t i = 0; i < l.owned_faces; ++i)
      before_u[r][i] = initial[l.faces[i]];
    for (size_t i = 0; i < l.owned_cells; ++i)
      before_p[r][i] = -.321;
    if (!buffers.Allocate(devices[r], before_u[r].size(), u[r]) ||
        !buffers.Allocate(devices[r], before_p[r].size(), p[r]) ||
        cudaMemcpy(u[r], before_u[r].data(), before_u[r].size() * sizeof(double), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(p[r], before_p[r].data(), before_p[r].size() * sizeof(double), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess)
      return 6;
  }
  auto unchanged = [&] {
    for (size_t r = 0; r < devices.size(); ++r) {
      cudaSetDevice(devices[r]);
      std::vector<double> a(before_u[r].size()), b(before_p[r].size());
      if (cudaMemcpy(a.data(), u[r], a.size() * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess ||
          cudaMemcpy(b.data(), p[r], b.size() * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess ||
          a != before_u[r] || b != before_p[r])
        return false;
    }
    return true;
  };
  if (gpu.Advance(u, p, -.01, 1).advanced || !unchanged())
    return 7;
  auto failed = gpu.Advance(u, p, .01, 1, true, 1.225, 9.2e-7, 1.5e-5, .4 * width, 1e-14, 1);
  if (failed.advanced || failed.projection.converged || failed.projection.iterations != 1 || gpu.Error().empty() ||
      !unchanged())
    return 8;
  if (body) {
    failed = gpu.Advance(u, p, .01, 1, true, 1.225, 1e300, 1.5e-5, .4 * width, 1e-10);
    if (failed.advanced || !failed.projection.converged || gpu.Error().empty() || !unchanged())
      return 9;
  }
  double max_u = 0, max_p = 0, max_load = 0;
  for (int step = 0; step < 3; ++step) {
    double dt = .005 * (step + 1), speed = step == 0 ? 1. : step == 1 ? .7 : -.3;
    bool cubic = step != 1;
    auto reference = single.AdvanceDevice(projection, single_u, initial.size(), single_p, expected_p.size(), dt, speed,
                                          cubic, 1.225, 9.2e-7, 1.5e-5, .4 * width, 1e-10);
    cudaSetDevice(devices.back());
    auto actual = gpu.Advance(u, p, dt, speed, cubic, 1.225, 9.2e-7, 1.5e-5, .4 * width, 1e-10);
    int current = -1;
    cudaGetDevice(&current);
    if (!reference.advanced || !actual.advanced || current != devices.back()) {
      std::cerr << gpu.Error() << '\n';
      return 10;
    }
    if (!std::isfinite(actual.transport_seconds + actual.pressure_seconds + actual.loads_seconds) ||
        actual.transport_seconds < 0 || actual.pressure_seconds < 0 || actual.loads_seconds < 0)
      return 11;
    cudaSetDevice(devices[0]);
    if (cudaMemcpy(expected_u.data(), single_u, expected_u.size() * sizeof(double), cudaMemcpyDeviceToHost) !=
            cudaSuccess ||
        cudaMemcpy(expected_p.data(), single_p, expected_p.size() * sizeof(double), cudaMemcpyDeviceToHost) !=
            cudaSuccess)
      return 12;
    for (size_t r = 0; r < devices.size(); ++r) {
      auto &l = layouts[r];
      std::vector<double> a(before_u[r].size()), b(before_p[r].size());
      cudaSetDevice(devices[r]);
      if (cudaMemcpy(a.data(), u[r], a.size() * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess ||
          cudaMemcpy(b.data(), p[r], b.size() * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess)
        return 13;
      for (size_t i = 0; i < l.owned_faces; ++i) {
        if (!std::isfinite(a[i]))
          return 14;
        max_u = std::max(max_u, std::abs(a[i] - expected_u[l.faces[i]]));
      }
      for (size_t i = 0; i < l.owned_cells; ++i) {
        if (!std::isfinite(b[i]))
          return 15;
        max_p = std::max(max_p, std::abs(b[i] - expected_p[l.cells[i]]));
      }
      for (size_t i = l.owned_faces; i < a.size(); ++i)
        if (a[i] != -54321.)
          return 16;
      for (size_t i = l.owned_cells; i < b.size(); ++i)
        if (b[i] != -54321.)
          return 17;
    }
    for (int k = 0; k < 6; ++k) {
      double pe = std::abs(actual.pressure_loads[k] - reference.pressure_loads[k]);
      double fe = std::abs(actual.friction_loads[k] - reference.friction_loads[k]);
      max_load = std::max({max_load, pe, fe});
      if (pe > 1e-7 * (1 + std::abs(reference.pressure_loads[k])) ||
          fe > 1e-8 + 2e-6 * std::abs(reference.friction_loads[k]) ||
          actual.total_loads[k] != actual.pressure_loads[k] + actual.friction_loads[k])
        return 18;
    }
    if (max_u > 1e-7 || max_p > 1e-6) {
      std::cerr << "u=" << max_u << " p=" << max_p << '\n';
      return 19;
    }
  }
  std::cout << "PASS distributed advance ranks=" << devices.size() << " mixed=" << mixed << " body=" << body
            << " width=" << width << " velocity_error=" << max_u << " pressure_error=" << max_p
            << " load_error=" << max_load << '\n';
  return 0;
}
}  // namespace

int main() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2)
    return 77;
  std::vector<std::vector<int>> mappings{{0}, {0, 1}, {1, 0}};
  if (count >= 4)
    mappings.push_back({3, 1, 0, 2});
  for (bool mixed : {false, true})
    for (bool body : {false, true})
      for (double width : {1., 1.3})
        for (auto devices : mappings)
          if (int status = Run(mixed, body, width, devices)) {
            std::cerr << "advance failure " << status << '\n';
            return status;
          }
}
