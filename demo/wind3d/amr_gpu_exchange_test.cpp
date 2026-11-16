#include "amr_gpu_exchange.h"

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <limits>

namespace {
struct Buffers {
  std::vector<int> devices;
  std::vector<double *> cells, faces;

  ~Buffers() {
    int previous = 0;
    cudaGetDevice(&previous);
    for (size_t r = 0; r < devices.size(); ++r) {
      cudaSetDevice(devices[r]);
      cudaFree(cells[r]);
      cudaFree(faces[r]);
    }
    cudaSetDevice(previous);
  }
};
}  // namespace

int Run(bool mixed, std::vector<int> devices) {
  using namespace wind3d::amr;
  Topology tree;
  Mesh mesh;
  FaceMass mass;
  PressurePartition plan;
  if (!tree.Regrid({2, 2, 2}, 1, 64, [mixed](Block b) { return !mixed || (b.x == 0 && b.y == 0); }) ||
      !mesh.Build(tree, 4, 1.3, {-1.3, -1.3, -1.3}) || !mass.Build(mesh) ||
      !plan.Build(mesh, tree, mass, int(devices.size())))
    return 1;
  GpuPressureExchange exchange;
  int previous = devices.back();
  if (cudaSetDevice(previous) != cudaSuccess || !exchange.Build(plan, devices)) {
    std::cerr << exchange.Error() << '\n';
    return 2;
  }
  int current = -1;
  cudaGetDevice(&current);
  if (current != previous)
    return 3;
  const auto &layouts = exchange.Layouts();
  if (exchange.Build(plan, {}) || exchange.Error().empty() || layouts.size() != devices.size())
    return 4;
  auto invalid = devices;
  invalid[0] = -1;
  if (exchange.Build(plan, invalid))
    return 5;
  if (devices.size() > 1 && exchange.Build(plan, std::vector<int>(devices.size(), devices[0])))
    return 6;
  Buffers values;
  values.devices = devices;
  values.cells.resize(devices.size());
  values.faces.resize(devices.size());
  for (size_t r = 0; r < devices.size(); ++r) {
    if (cudaSetDevice(devices[r]) != cudaSuccess ||
        cudaMalloc(&values.cells[r], layouts[r].cells.size() * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&values.faces[r], layouts[r].faces.size() * sizeof(double)) != cudaSuccess)
      return 7;
  }
  for (int repeat = 0; repeat < 8; ++repeat) {
    for (bool faces : {false, true}) {
      auto &buffers = faces ? values.faces : values.cells;
      for (size_t r = 0; r < devices.size(); ++r) {
        const auto &ids = faces ? layouts[r].faces : layouts[r].cells;
        size_t owned = faces ? layouts[r].owned_faces : layouts[r].owned_cells;
        std::vector<double> host(ids.size(), std::numeric_limits<double>::quiet_NaN());
        for (size_t i = 0; i < owned; ++i)
          host[i] = ids[i] * 3. + repeat;
        if (cudaSetDevice(devices[r]) != cudaSuccess ||
            cudaMemcpy(buffers[r], host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess)
          return 8;
      }
      if (cudaSetDevice(previous) != cudaSuccess)
        return 9;
      if (!(faces ? exchange.ExchangeFaces(buffers) : exchange.ExchangeCells(buffers))) {
        std::cerr << exchange.Error() << '\n';
        return 10;
      }
      cudaGetDevice(&current);
      if (current != previous)
        return 11;
      // Rejected input must not alter either owned data or completed imports.
      if (exchange.ExchangeCells({}) || exchange.ExchangeFaces(std::vector<double *>(devices.size(), nullptr)))
        return 12;
      for (size_t r = 0; r < devices.size(); ++r) {
        const auto &ids = faces ? layouts[r].faces : layouts[r].cells;
        std::vector<double> actual(ids.size());
        if (cudaSetDevice(devices[r]) != cudaSuccess ||
            cudaMemcpy(actual.data(), buffers[r], actual.size() * sizeof(double), cudaMemcpyDeviceToHost) !=
                cudaSuccess)
          return 13;
        for (size_t i = 0; i < ids.size(); ++i)
          if (actual[i] != ids[i] * 3. + repeat)
            return 14;
      }
    }
  }
  std::cout << "PASS peer exchange mixed=" << mixed << " ranks=" << devices.size() << '\n';
  return 0;
}

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2)
    return 77;
  for (bool mixed : {false, true})
    for (auto mapping : {std::vector<int>{0}, std::vector<int>{0, 1}, std::vector<int>{1, 0}})
      if (int status = Run(mixed, mapping)) {
        std::cerr << "exchange failure " << status << '\n';
        return status;
      }
}
