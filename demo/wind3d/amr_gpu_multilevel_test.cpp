#include "amr_gpu_multilevel.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>

#include "amr_partition.h"

using namespace wind3d::amr;

std::vector<double> Product(const PressureLevel &a, const std::vector<double> &x) {
  std::vector<double> y(x.size());
  for (size_t i = 0; i < y.size(); ++i)
    for (int k = a.offsets[i]; k < a.offsets[i + 1]; ++k)
      y[i] += a.weights[k] * x[a.columns[k]];
  return y;
}

std::vector<double> Cycle(const std::vector<PressureLevel> &levels, size_t d, const std::vector<double> &b) {
  const auto &a = levels[d];
  size_t n = b.size();
  if (a.identity)
    return b;
  if (d + 1 == levels.size()) {
    // Independent Cholesky solve, rather than the device's stored dense inverse.
    std::vector<double> l(n * n), x = b;
    for (size_t i = 0; i < n; ++i)
      for (int k = a.offsets[i]; k < a.offsets[i + 1]; ++k)
        l[i * n + a.columns[k]] = a.weights[k];
    for (size_t i = 0; i < n; ++i)
      for (size_t j = 0; j <= i; ++j) {
        double v = l[i * n + j];
        for (size_t k = 0; k < j; ++k)
          v -= l[i * n + k] * l[j * n + k];
        l[i * n + j] = i == j ? std::sqrt(v) : v / l[j * n + j];
      }
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < i; ++j)
        x[i] -= l[i * n + j] * x[j];
      x[i] /= l[i * n + i];
    }
    for (size_t i = n; i-- > 0;) {
      for (size_t j = i + 1; j < n; ++j)
        x[i] -= l[j * n + i] * x[j];
      x[i] /= l[i * n + i];
    }
    return x;
  }
  std::vector<double> x(n);
  auto smooth = [&]() {
    auto ax = Product(a, x);
    for (size_t i = 0; i < n; ++i)
      x[i] += .9 * (b[i] - ax[i]) / a.diagonal[i];
  };
  smooth();
  smooth();
  auto ax = Product(a, x);
  std::vector<double> coarse(levels[d + 1].owners.size());
  for (size_t i = 0; i < n; ++i)
    if (a.parent[i] >= 0)
      coarse[a.parent[i]] += b[i] - ax[i];
  auto correction = Cycle(levels, d + 1, coarse);
  for (size_t i = 0; i < n; ++i)
    if (a.parent[i] >= 0)
      x[i] += correction[a.parent[i]];
  smooth();
  smooth();
  return x;
}

struct Buffers {
  std::vector<std::pair<int, double *>> entries;

  ~Buffers() {
    for (auto [d, p] : entries) {
      cudaSetDevice(d);
      cudaFree(p);
    }
  }

  bool Allocate(int d, size_t n, double *&p) {
    if (cudaSetDevice(d) != cudaSuccess || cudaMalloc(&p, n * sizeof(double)) != cudaSuccess)
      return false;
    entries.push_back({d, p});
    return true;
  }
};

int Run(bool mixed, int geometry, const std::vector<int> &devices) {
  Topology tree;
  Mesh mesh;
  FaceMass mass;
  PressurePartition partition;
  PressureHierarchy hierarchy;
  GpuMultilevel gpu;
  if (!tree.Regrid({2, 2, 2}, 1, 64, [=](Block b) { return !mixed || (b.x == 0 && b.y == 0); }) ||
      !mesh.Build(tree, 2, 1.3, {-1.3, -1.3, -1.3}))
    return 1;
  if (geometry && !mesh.Embed([=](std::array<double, 3> p) {
        if (geometry == 2)
          return -1.;
        return std::sqrt((p[0] + .65) * (p[0] + .65) + (p[1] + .65) * (p[1] + .65) + (p[2] + .65) * (p[2] + .65)) - .2;
      }))
    return 2;
  if (!mass.Build(mesh) || !partition.Build(mesh, tree, mass, int(devices.size())) ||
      !hierarchy.Build(mesh, partition.CellOwners(), int(devices.size()), 1) || !gpu.Build(hierarchy, devices)) {
    std::cerr << gpu.Error() << '\n';
    return 3;
  }
  if (gpu.Build(hierarchy, {}) || gpu.Build(hierarchy, {0, 0}))
    return 4;
  auto &fine = hierarchy.Levels().front();
  size_t n = mesh.Cells().size();
  Buffers storage;
  std::vector<double *> input(devices.size()), output(devices.size());
  for (size_t r = 0; r < devices.size(); ++r)
    if (!storage.Allocate(devices[r], fine.owned_cells[r].size() + 3, input[r]) ||
        !storage.Allocate(devices[r], fine.owned_cells[r].size() + 3, output[r]))
      return 5;
  std::vector<std::vector<double>> b(3, std::vector<double>(n)), solutions;
  for (size_t i = 0; i < n; ++i) {
    b[0][i] = std::sin(.17 * i);
    b[1][i] = std::cos(.31 * i);
    b[2][i] = b[0][i] - .7 * b[1][i];
  }
  double maximum = 0;
  for (int repeat = 0; repeat < 4; ++repeat) {
    int sample = repeat % 3;
    auto reference = Cycle(hierarchy.Levels(), 0, b[sample]);
    std::vector<double> assembled(n);
    for (size_t r = 0; r < devices.size(); ++r) {
      const auto &ids = fine.owned_cells[r];
      std::vector<double> host(ids.size() + 3, 123);
      for (size_t i = 0; i < ids.size(); ++i)
        host[i] = b[sample][ids[i]];
      if (cudaSetDevice(devices[r]) != cudaSuccess ||
          cudaMemcpy(input[r], host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess)
        return 6;
      std::fill(host.begin(), host.end(), -777);
      if (cudaMemcpy(output[r], host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
          cudaDeviceSynchronize() != cudaSuccess)
        return 7;
    }
    int saved = devices.back(), current = -1;
    cudaSetDevice(saved);
    if (!gpu.Apply(input, output)) {
      std::cerr << gpu.Error() << '\n';
      return 8;
    }
    cudaGetDevice(&current);
    if (current != saved || gpu.Apply(input, input))
      return 9;
    for (size_t r = 0; r < devices.size(); ++r) {
      const auto &ids = fine.owned_cells[r];
      std::vector<double> host(ids.size() + 3);
      if (cudaSetDevice(devices[r]) != cudaSuccess ||
          cudaMemcpy(host.data(), output[r], host.size() * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess)
        return 10;
      for (size_t i = 0; i < ids.size(); ++i) {
        double error = std::abs(host[i] - reference[ids[i]]);
        maximum = std::max(maximum, error);
        if (!std::isfinite(host[i]) || error > 2e-10 * std::max(1., std::abs(reference[ids[i]])))
          return 11;
        assembled[ids[i]] = host[i];
      }
      for (size_t i = ids.size(); i < host.size(); ++i)
        if (host[i] != -777)
          return 12;
    }
    solutions.push_back(assembled);
  }
  double xy = 0, yx = 0, energy = 0;
  for (size_t i = 0; i < n; ++i) {
    xy += b[0][i] * solutions[1][i];
    yx += b[1][i] * solutions[0][i];
    energy += b[0][i] * solutions[0][i];
    if (std::abs(solutions[2][i] - (solutions[0][i] - .7 * solutions[1][i])) > 1e-9 ||
        std::abs(solutions[0][i] - solutions[3][i]) > 1e-12)
      return 13;
  }
  if (!(energy > 0) || std::abs(xy - yx) > 1e-10 * std::max({1., std::abs(xy), std::abs(yx)}))
    return 14;
  std::cout << "PASS GPU V-cycle mixed=" << mixed << " geometry=" << geometry << " ranks=" << devices.size()
            << " error=" << maximum << '\n';
  return 0;
}

int main() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2)
    return 77;
  for (bool mixed : {false, true})
    for (int geometry : {0, 1, 2})
      if (!(mixed && geometry == 2))
        for (auto devices : {std::vector<int>{0}, std::vector<int>{0, 1}, std::vector<int>{1, 0}})
          if (int result = Run(mixed, geometry, devices)) {
            std::cerr << "V-cycle failure " << result << '\n';
            return result;
          }
}
