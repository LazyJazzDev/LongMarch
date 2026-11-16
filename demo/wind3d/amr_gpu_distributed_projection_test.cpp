#include "amr_gpu_distributed_projection.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

#include "amr_gpu_projection.h"

struct Buffers {
  std::vector<std::pair<int, double *>> allocations;

  ~Buffers() {
    int old = 0;
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

int Run(bool mixed, bool embedded, double width, const std::vector<int> &devices, bool multilevel) {
  using namespace wind3d::amr;
  Topology tree;
  Mesh mesh;
  FaceGradient gradient;
  if (!tree.Regrid({2, 2, 2}, 1, 64, [mixed](Block b) { return !mixed || (b.x == 0 && b.y == 0); }) ||
      !mesh.Build(tree, 2, width, {-width, -width, -width}))
    return 1;
  if (embedded && !mesh.Embed([width](std::array<double, 3> p) {
        return std::max({std::abs(p[0] + .5 * width), std::abs(p[1] + .5 * width), std::abs(p[2] + .5 * width)}) -
               .22 * width;
      }))
    return 2;
  GpuProjection single;
  GpuDistributedProjection gpu;
  if (!single.Build(0, mesh, gradient, false, true, true) || !gpu.Build(mesh, tree, devices, multilevel)) {
    std::cerr << single.Error() << ' ' << gpu.Error() << '\n';
    return 3;
  }
  // A second solver may share already-enabled peer access. Its destruction and
  // a failed rebuild must not disrupt the first solver's subsequent projections.
  if (devices.size() > 1) {
    {
      GpuDistributedProjection second;
      if (!second.Build(mesh, tree, devices, multilevel))
        return 26;
      if (second.Build(mesh, tree, {devices[0], devices[0]}, multilevel))
        return 27;
    }
    for (int source : devices)
      for (int destination : devices)
        if (source != destination) {
          if (cudaSetDevice(source) != cudaSuccess)
            return 28;
          auto status = cudaDeviceEnablePeerAccess(destination, 0);
          if (status != cudaErrorPeerAccessAlreadyEnabled)
            return 29;
          cudaGetLastError();
        }
  }
  const auto &layouts = gpu.Layouts();
  Mesh empty;
  if (gpu.Build(empty, tree, devices) || gpu.Error().empty())
    return 4;
  std::vector<double> original(mesh.Faces().size());
  for (size_t i = 0; i < original.size(); ++i)
    original[i] = .2 + std::sin(.13 * i);
  auto reference = original;
  std::vector<double> reference_p;
  auto single_result = single.Project(reference, reference_p, 1e-10, 2000);
  if (!single_result.converged) {
    std::cerr << single.Error() << '\n';
    return 5;
  }
  Buffers storage;
  std::vector<double *> velocity(devices.size()), potential(devices.size());
  std::vector<std::vector<double>> before_u(devices.size()), before_p(devices.size());
  for (size_t r = 0; r < devices.size(); ++r) {
    auto &l = layouts[r];
    before_u[r].assign(l.faces.size(), 777);
    before_p[r].assign(l.cells.size(), -555);
    for (size_t i = 0; i < l.owned_faces; ++i)
      before_u[r][i] = original[l.faces[i]];
    if (!storage.Allocate(devices[r], l.faces.size(), velocity[r]) ||
        !storage.Allocate(devices[r], l.cells.size(), potential[r]) ||
        cudaMemcpy(velocity[r], before_u[r].data(), l.faces.size() * sizeof(double), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(potential[r], before_p[r].data(), l.cells.size() * sizeof(double), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess)
      return 6;
  }
  if (gpu.Project(velocity, potential, 1e-14, 1).converged || gpu.Project(velocity, potential, 0).converged ||
      gpu.Project(velocity, velocity).converged)
    return 7;
  for (size_t r = 0; r < devices.size(); ++r) {
    auto u = before_u[r], p = before_p[r];
    if (cudaSetDevice(devices[r]) != cudaSuccess ||
        cudaMemcpy(u.data(), velocity[r], u.size() * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(p.data(), potential[r], p.size() * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess ||
        u != before_u[r] || p != before_p[r])
      return 8;
  }
  int saved = devices.back(), current = -1;
  if (cudaSetDevice(saved) != cudaSuccess)
    return 9;
  auto result = gpu.Project(velocity, potential, 1e-10, 2000);
  cudaGetDevice(&current);
  if (!result.converged || result.relative_residual > 1e-10 || current != saved) {
    std::cerr << gpu.Error() << " residual=" << result.relative_residual << '\n';
    return 10;
  }
  std::vector<double> assembled(original.size());
  double error = 0, pressure_error = 0;
  for (size_t r = 0; r < devices.size(); ++r) {
    auto u = before_u[r], p = before_p[r];
    auto &l = layouts[r];
    if (cudaSetDevice(devices[r]) != cudaSuccess ||
        cudaMemcpy(u.data(), velocity[r], u.size() * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(p.data(), potential[r], p.size() * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess)
      return 11;
    for (size_t i = 0; i < u.size(); ++i) {
      if (!std::isfinite(u[i]))
        return 21;
      if (i < l.owned_faces) {
        int id = l.faces[i];
        assembled[id] = u[i];
        error = std::max(error, std::abs(u[i] - reference[id]));
        auto f = mesh.Faces()[id];
        if (f.aperture == 0 && u[i] != 0)
          return 12;
        if (f.aperture > 0 && (f.low < 0 || f.high < 0) && !(f.axis == 0 && f.high < 0) && u[i] != original[id])
          return 13;
      } else if (u[i] != 777)
        return 14;
    }
    for (size_t i = 0; i < p.size(); ++i)
      if (i < l.owned_cells)
        pressure_error = std::max(pressure_error, std::abs(p[i] - reference_p[l.cells[i]]));
      else if (p[i] != -555)
        return 15;
  }
  double divergence2 = 0, initial2 = 0;
  for (size_t i = 0; i < mesh.Cells().size(); ++i) {
    double div = 0, initial = 0;
    for (int k = mesh.Offsets()[i]; k < mesh.Offsets()[i + 1]; ++k) {
      int e = mesh.Incidence()[k], j = std::abs(e) - 1;
      auto f = mesh.Faces()[j];
      double w = (e > 0 ? -1 : 1) * f.area * f.aperture;
      div += w * assembled[j];
      initial += w * original[j];
    }
    divergence2 += div * div;
    initial2 += initial * initial;
  }
  double relative = std::sqrt(divergence2 / initial2);
  if (!std::isfinite(relative) || relative > 1.1e-10 || error > 1e-8 || pressure_error > 1e-8)
    return 16;
  // Reuse the solver after restoring the original input. Pageable host uploads
  // can return before DMA completes; finish producers before private streams run.
  for (int repeat = 0; repeat < 3; ++repeat) {
    for (size_t r = 0; r < devices.size(); ++r) {
      if (cudaSetDevice(devices[r]) != cudaSuccess ||
          cudaMemcpy(velocity[r], before_u[r].data(), before_u[r].size() * sizeof(double), cudaMemcpyHostToDevice) !=
              cudaSuccess ||
          cudaDeviceSynchronize() != cudaSuccess)
        return 21;
    }
    auto repeated = gpu.Project(velocity, potential, 1e-10, 2000);
    if (!repeated.converged || repeated.relative_residual > 1e-10)
      return 22;
    for (size_t r = 0; r < devices.size(); ++r) {
      auto values = before_u[r];
      if (cudaSetDevice(devices[r]) != cudaSuccess ||
          cudaMemcpy(values.data(), velocity[r], values.size() * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess)
        return 23;
      for (size_t i = 0; i < layouts[r].owned_faces; ++i)
        if (!std::isfinite(values[i]) || std::abs(values[i] - assembled[layouts[r].faces[i]]) > 1e-10)
          return 24;
      for (size_t i = layouts[r].owned_faces; i < values.size(); ++i)
        if (values[i] != before_u[r][i])
          return 25;
    }
  }
  // A zero right-hand side must succeed and replace the previous potential.
  for (size_t r = 0; r < devices.size(); ++r) {
    auto &l = layouts[r];
    std::vector<double> zero(l.faces.size());
    if (cudaSetDevice(devices[r]) != cudaSuccess ||
        cudaMemcpy(velocity[r], zero.data(), zero.size() * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess)
      return 17;
  }
  auto zero_result = gpu.Project(velocity, potential);
  if (!zero_result.converged || zero_result.iterations != 0)
    return 18;
  for (size_t r = 0; r < devices.size(); ++r) {
    std::vector<double> p(layouts[r].cells.size());
    if (cudaSetDevice(devices[r]) != cudaSuccess ||
        cudaMemcpy(p.data(), potential[r], p.size() * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess)
      return 19;
    for (size_t i = 0; i < layouts[r].owned_cells; ++i)
      if (p[i] != 0)
        return 20;
  }
  std::cout << "PASS distributed projection mixed=" << mixed << " embedded=" << embedded << " width=" << width
            << " multilevel=" << multilevel << " ranks=" << devices.size() << " iterations=" << result.iterations
            << " velocity_error=" << error << " pressure_error=" << pressure_error << " divergence=" << relative
            << '\n';
  return 0;
}

int main() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n < 2)
    return 77;
  for (bool mixed : {false, true})
    for (bool embedded : {false, true})
      for (double width : {1., 1.3})
        for (auto devices : {std::vector<int>{0}, std::vector<int>{0, 1}, std::vector<int>{1, 0}})
          for (bool multilevel : {false, true})
            if (int status = Run(mixed, embedded, width, devices, multilevel)) {
              std::cerr << "distributed projection failure " << status << '\n';
              return status;
            }
}
