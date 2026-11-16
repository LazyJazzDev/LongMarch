#include "amr_gpu_operators.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

#include "amr_gpu_mass.h"

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

int Run(bool mixed, bool embedded, double width, const std::vector<int> &devices, bool pressure_boundaries) {
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
  if (!gpu.Build(mesh, tree, devices, pressure_boundaries)) {
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
  GpuMassInverse single;
  double *full_rhs = nullptr, *full_result = nullptr;
  std::vector<double> inverse(u.size()), assembled(u.size());
  if (!single.Build(0, mesh, pressure_boundaries) || !storage.Allocate(0, u.size(), full_rhs) ||
      !storage.Allocate(0, u.size(), full_result) ||
      cudaMemcpy(full_rhs, u.data(), u.size() * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
      !single.ApplyDevice(full_rhs, full_result, u.size()) ||
      cudaMemcpy(inverse.data(), full_result, u.size() * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess)
    return 15;
  std::vector<double> gradient(u.size()), pressure(p.size());
  for (size_t i = 0; i < u.size(); ++i) {
    auto f = mesh.Faces()[i];
    gradient[i] = ((f.low >= 0 ? -p[f.low] : 0) + (f.high >= 0 ? p[f.high] : 0)) * f.area * f.aperture;
  }
  if (cudaSetDevice(0) != cudaSuccess ||
      cudaMemcpy(full_rhs, gradient.data(), u.size() * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
      !single.ApplyDevice(full_rhs, full_result, u.size()) ||
      cudaMemcpy(gradient.data(), full_result, u.size() * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess)
    return 18;
  for (size_t i = 0; i < p.size(); ++i) {
    for (int k = mesh.Offsets()[i]; k < mesh.Offsets()[i + 1]; ++k) {
      int e = mesh.Incidence()[k], j = std::abs(e) - 1;
      auto f = mesh.Faces()[j];
      pressure[i] += (e > 0 ? -1 : 1) * f.area * f.aperture * gradient[j];
    }
    if (!mesh.Cells()[i].active)
      pressure[i] = p[i];
  }
  auto free_face = [&](size_t i) {
    const auto &f = mesh.Faces()[i];
    return f.aperture > 0 && (!pressure_boundaries || (f.low >= 0 && f.high >= 0) || (f.axis == 0 && f.high < 0));
  };
  double maximum = 0;
  for (int operation = 0; operation < 8; ++operation) {
    double factor = operation == 4 ? -.5 : 1;
    bool cell_input = operation == 1 || operation >= 5, cell_output = operation == 0 || operation >= 6;
    auto &input = cell_input ? cells : faces;
    auto &output = cell_output ? cell_out : face_out;
    for (size_t r = 0; r < devices.size(); ++r) {
      auto &l = layouts[r];
      const auto &ids = cell_input ? l.cells : l.faces;
      size_t owned = cell_input ? l.owned_cells : l.owned_faces;
      std::vector<double> host(ids.size(), std::numeric_limits<double>::quiet_NaN());
      for (size_t i = 0; i < owned; ++i)
        host[i] = cell_input ? p[ids[i]] : factor * u[ids[i]];
      if (cudaSetDevice(devices[r]) != cudaSuccess ||
          cudaMemcpy(input[r], host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess)
        return 7;
      host.assign(cell_output ? l.cells.size() : l.faces.size(), -999);
      if (cudaMemcpy(output[r], host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
          cudaDeviceSynchronize() != cudaSuccess)
        return 8;
    }
    int saved = devices.back(), actual = -1;
    if (cudaSetDevice(saved) != cudaSuccess)
      return 9;
    if (!(operation == 0   ? gpu.Divergence(input, output)
          : operation == 1 ? gpu.TransposeDivergence(input, output)
          : operation == 2 ? gpu.Mass(input, output)
          : operation < 5  ? gpu.InverseMass(input, output)
          : operation == 5 ? gpu.Gradient(input, output)
          : operation == 6 ? gpu.Pressure(input, output)
                           : gpu.TwoPointPressure(input, output))) {
      std::cerr << gpu.Error() << '\n';
      return 10;
    }
    cudaGetDevice(&actual);
    if (actual != saved)
      return 11;
    if (gpu.TwoPointPressure(input, input) || gpu.Gradient(input, input) || gpu.Pressure(input, input) ||
        gpu.InverseMass(input, input) || gpu.Mass(input, input) || gpu.Divergence({}, output) ||
        gpu.TransposeDivergence(input, {}))
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
          } else if (operation == 2) {
            for (int k = mass.Offsets()[id]; k < mass.Offsets()[id + 1]; ++k)
              reference += mass.Weights()[k] * u[mass.Columns()[k]];
          } else if (operation < 5) {
            reference = factor * inverse[id];
            assembled[id] = host[i];
          } else if (operation == 5)
            reference = gradient[id];
          else if (operation == 6)
            reference = pressure[id];
          else {
            for (int k = mesh.Offsets()[id]; k < mesh.Offsets()[id + 1]; ++k) {
              int j = std::abs(mesh.Incidence()[k]) - 1;
              auto f = mesh.Faces()[j];
              if (!free_face(j))
                continue;
              int neighbor = f.low == id ? f.high : f.low;
              reference += f.area * f.aperture / f.distance * (p[id] - (neighbor >= 0 ? p[neighbor] : 0));
            }
            if (!mesh.Cells()[id].active)
              reference = p[id];
          }
        }
        if (!std::isfinite(host[i]) || std::abs(host[i] - reference) > 1e-12 * std::max(1., std::abs(reference)))
          return 14;
        maximum = std::max(maximum, std::abs(host[i] - reference));
      }
    }
    if (operation == 3 || operation == 4) {
      double residual = 0;
      for (size_t i = 0; i < u.size(); ++i) {
        if (!free_face(i)) {
          if (assembled[i] != 0)
            return 16;
          continue;
        }
        double value = 0;
        for (int k = mass.Offsets()[i]; k < mass.Offsets()[i + 1]; ++k)
          value += mass.Weights()[k] * assembled[mass.Columns()[k]];
        residual = std::max(residual, std::abs(value - factor * u[i]));
      }
      if (residual > 1e-12) {
        std::cerr << "true mass residual " << residual << '\n';
        return 17;
      }
    }
  }
  std::cout << "PASS distributed D/DT/M/inverse/gradient/pressure ranks=" << devices.size() << " mixed=" << mixed
            << " embedded=" << embedded << " width=" << width << " error=" << maximum << '\n';
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
          for (bool boundaries : {false, true})
            if (int status = Run(mixed, embedded, width, devices, boundaries)) {
              std::cerr << "operator failure " << status << '\n';
              return status;
            }
}
