#include "amr_gpu_operators.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

struct Buffers {
  std::vector<std::pair<int, double *>> allocations;

  ~Buffers() {
    int old = 0;
    cudaGetDevice(&old);
    for (auto [device, p] : allocations) {
      cudaSetDevice(device);
      cudaFree(p);
    }
    cudaSetDevice(old);
  }

  bool Allocate(int device, size_t n, double *&p) {
    if (cudaSetDevice(device) != cudaSuccess || cudaMalloc(&p, n * sizeof(double)) != cudaSuccess)
      return false;
    allocations.push_back({device, p});
    return true;
  }
};

int Run(bool mixed, bool embedded, double width, const std::vector<int> &devices) {
  using namespace wind3d::amr;
  Topology tree;
  Mesh mesh;
  FaceMass mass;
  if (!tree.Regrid({2, 2, 2}, 1, 64, [mixed](Block b) { return !mixed || (b.x == 0 && b.y == 0); }) ||
      !mesh.Build(tree, 4, width, {-width, -width, -width}))
    return 1;
  if (embedded && !mesh.Embed([width](std::array<double, 3> p) {
        return std::sqrt((p[0] + .5 * width) * (p[0] + .5 * width) + (p[1] + .5 * width) * (p[1] + .5 * width) +
                         (p[2] + .5 * width) * (p[2] + .5 * width)) -
               .18 * width;
      }))
    return 2;
  if (!mass.Build(mesh))
    return 3;
  GpuPressureOperators gpu;
  if (!gpu.Build(mesh, tree, devices)) {
    std::cerr << gpu.Error() << '\n';
    return 4;
  }
  const auto &layouts = gpu.Layouts();
  if (gpu.Build(mesh, tree, {}) || layouts.size() != devices.size())
    return 5;
  Buffers storage;
  std::vector<double *> cells(devices.size()), faces(devices.size()), cell_out(devices.size()),
      face_out(devices.size());
  for (size_t r = 0; r < devices.size(); ++r) {
    auto &l = layouts[r];
    if (!storage.Allocate(devices[r], l.cells.size(), cells[r]) ||
        !storage.Allocate(devices[r], l.faces.size(), faces[r]) ||
        !storage.Allocate(devices[r], l.cells.size(), cell_out[r]) ||
        !storage.Allocate(devices[r], l.faces.size(), face_out[r]))
      return 6;
  }
  std::vector<double> p(mesh.Cells().size()), u(mesh.Faces().size());
  for (size_t i = 0; i < p.size(); ++i)
    p[i] = std::cos(.21 * i);
  for (size_t i = 0; i < u.size(); ++i)
    u[i] = std::sin(.17 * i);
  double maximum = 0;
  for (int operation = 0; operation < 3; ++operation) {
    bool cell_input = operation == 1, cell_output = operation == 0;
    auto &input = cell_input ? cells : faces;
    auto &output = cell_output ? cell_out : face_out;
    for (size_t r = 0; r < devices.size(); ++r) {
      auto &l = layouts[r];
      const auto &ids = cell_input ? l.cells : l.faces;
      size_t owned = cell_input ? l.owned_cells : l.owned_faces;
      std::vector<double> host(ids.size(), std::numeric_limits<double>::quiet_NaN());
      for (size_t i = 0; i < owned; ++i)
        host[i] = cell_input ? p[ids[i]] : u[ids[i]];
      if (cudaSetDevice(devices[r]) != cudaSuccess ||
          cudaMemcpy(input[r], host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess)
        return 7;
      host.assign(cell_output ? l.cells.size() : l.faces.size(), -999);
      if (cudaMemcpy(output[r], host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess)
        return 8;
    }
    int saved = devices.back(), actual = -1;
    if (cudaSetDevice(saved) != cudaSuccess)
      return 9;
    if (!(operation == 0   ? gpu.Divergence(input, output)
          : operation == 1 ? gpu.TransposeDivergence(input, output)
                           : gpu.Mass(input, output))) {
      std::cerr << gpu.Error() << '\n';
      return 10;
    }
    cudaGetDevice(&actual);
    if (actual != saved)
      return 11;
    if (gpu.Mass(input, input) || gpu.Divergence({}, output) || gpu.TransposeDivergence(input, {}))
      return 12;
    for (size_t r = 0; r < devices.size(); ++r) {
      auto &l = layouts[r];
      const auto &ids = cell_output ? l.cells : l.faces;
      size_t owned = cell_output ? l.owned_cells : l.owned_faces;
      std::vector<double> host(ids.size());
      if (cudaSetDevice(devices[r]) != cudaSuccess ||
          cudaMemcpy(host.data(), output[r], host.size() * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess)
        return 13;
      for (size_t i = 0; i < host.size(); ++i) {
        double reference = -999;
        int id = ids[i];
        if (i < owned) {
          reference = 0;
          if (operation == 0) {
            for (int k = mesh.Offsets()[id]; k < mesh.Offsets()[id + 1]; ++k) {
              int e = mesh.Incidence()[k], j = std::abs(e) - 1;
              auto f = mesh.Faces()[j];
              reference += (e > 0 ? -1 : 1) * f.area * f.aperture * u[j];
            }
          } else if (operation == 1) {
            auto f = mesh.Faces()[id];
            reference = ((f.low >= 0 ? -p[f.low] : 0) + (f.high >= 0 ? p[f.high] : 0)) * f.area * f.aperture;
          } else {
            for (int k = mass.Offsets()[id]; k < mass.Offsets()[id + 1]; ++k)
              reference += mass.Weights()[k] * u[mass.Columns()[k]];
          }
        }
        if (!std::isfinite(host[i]) || std::abs(host[i] - reference) > 1e-12 * std::max(1., std::abs(reference)))
          return 14;
        maximum = std::max(maximum, std::abs(host[i] - reference));
      }
    }
  }
  std::cout << "PASS distributed D/DT/M ranks=" << devices.size() << " mixed=" << mixed << " embedded=" << embedded
            << " width=" << width << " error=" << maximum << '\n';
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
          if (int status = Run(mixed, embedded, width, devices)) {
            std::cerr << "operator failure " << status << '\n';
            return status;
          }
}
