#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

#include "amr_gpu_distributed_transport.h"
#include "amr_partition.h"
#include "amr_transport.h"
#include "amr_wall_loads.h"

namespace {
struct Buffers {
  std::vector<int> devices;
  std::vector<double *> velocity, pressure;

  ~Buffers() {
    int previous;
    cudaGetDevice(&previous);
    for (size_t r = 0; r < devices.size(); ++r) {
      cudaSetDevice(devices[r]);
      cudaFree(velocity[r]);
      cudaFree(pressure[r]);
    }
    cudaSetDevice(previous);
  }
};

bool Same(const wind3d::amr::TransportLoads &a, const wind3d::amr::TransportLoads &b) {
  return a.pressure == b.pressure && a.friction == b.friction && a.total == b.total;
}

int Run(bool mixed, int body, double width, const std::vector<int> &devices) {
  using namespace wind3d::amr;
  Topology tree;
  Mesh mesh;
  FaceMass mass;
  PressurePartition partition;
  if (!tree.Regrid({2, 2, 2}, 1, 100, [mixed](Block b) { return !mixed || (b.x == 0 && b.y == 0); }) ||
      !mesh.Build(tree, 4, width, {-width, -width, -width}))
    return 1;
  if (body && !mesh.Embed([=](std::array<double, 3> p) {
        return body == 2
                   ? -1.
                   : std::max({std::abs(p[0] + .5 * width), std::abs(p[1] + .5 * width), std::abs(p[2])}) - .21 * width;
      }))
    return 2;
  if (!mass.Build(mesh) || !partition.Build(mesh, tree, mass, int(devices.size())))
    return 3;
  GpuDistributedTransport gpu;
  int previous = devices.back(), current = -1;
  cudaSetDevice(previous);
  if (!gpu.Build(mesh, tree, devices, width, {-width, -width, -width})) {
    std::cerr << gpu.Error() << '\n';
    return 4;
  }
  if (gpu.OwnedFaces() != partition.OwnedFaces() || gpu.OwnedCells() != partition.OwnedCells())
    return 5;
  Buffers buffers;
  buffers.devices = devices;
  buffers.velocity.resize(devices.size());
  buffers.pressure.resize(devices.size());
  std::vector<const double *> up, pp;
  for (size_t r = 0; r < devices.size(); ++r) {
    cudaSetDevice(devices[r]);
    if (cudaMalloc(&buffers.velocity[r], (gpu.OwnedFaces()[r].size() + 3) * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&buffers.pressure[r], (gpu.OwnedCells()[r].size() + 3) * sizeof(double)) != cudaSuccess)
      return 6;
    up.push_back(buffers.velocity[r]);
    pp.push_back(buffers.pressure[r]);
  }
  double pressure_error = 0, friction_error = 0;
  TransportLoads last;
  for (int pattern = 0; pattern < 3; ++pattern)
    for (double sign : {-1., 1.}) {
      std::vector<double> velocity(mesh.Faces().size()), pressure(mesh.Cells().size());
      for (size_t i = 0; i < velocity.size(); ++i) {
        auto f = mesh.Faces()[i];
        velocity[i] = sign * (.3 + .2 * f.axis + .1 * std::sin(3 * f.centre[(f.axis + 1) % 3]));
      }
      for (size_t i = 0; i < pressure.size(); ++i) {
        auto p = mesh.Cells()[i].centre;
        pressure[i] = sign * (pattern == 0   ? 2.
                              : pattern == 1 ? 1. + p[0] + 2 * p[1] + .7 * p[2]
                                             : .3 * std::sin(p[0] - 2 * p[1] + p[2]));
      }
      double density = sign > 0 ? 1.225 : 2.4, viscosity = pattern == 2 ? 3e-6 : 9.2e-7;
      FaceField field(mesh, tree, width, {-width, -width, -width});
      auto extended = velocity;
      TransportLoads expected;
      if (!field.ExtendSlip(extended) || !mesh.PressureLoads(pressure, expected.pressure) ||
          !WallLoads(mesh, extended, density, viscosity, expected.friction))
        return 7;
      std::vector<std::vector<double>> local_u(devices.size()), local_p(devices.size());
      for (size_t r = 0; r < devices.size(); ++r) {
        auto &u = local_u[r], &p = local_p[r];
        u.resize(gpu.OwnedFaces()[r].size() + 3, std::numeric_limits<double>::quiet_NaN());
        p.resize(gpu.OwnedCells()[r].size() + 3, std::numeric_limits<double>::quiet_NaN());
        for (size_t i = 0; i < gpu.OwnedFaces()[r].size(); ++i)
          u[i] = velocity[gpu.OwnedFaces()[r][i]];
        for (size_t i = 0; i < gpu.OwnedCells()[r].size(); ++i)
          p[i] = pressure[gpu.OwnedCells()[r][i]];
        cudaSetDevice(devices[r]);
        if (cudaMemcpy(buffers.velocity[r], u.data(), u.size() * sizeof(double), cudaMemcpyHostToDevice) !=
                cudaSuccess ||
            cudaMemcpy(buffers.pressure[r], p.data(), p.size() * sizeof(double), cudaMemcpyHostToDevice) !=
                cudaSuccess ||
            cudaDeviceSynchronize() != cudaSuccess)
          return 8;
      }
      cudaSetDevice(previous);
      if (!gpu.EvaluateLoads(up, pp, last, density, viscosity)) {
        std::cerr << gpu.Error() << '\n';
        return 9;
      }
      cudaGetDevice(&current);
      if (current != previous)
        return 10;
      for (int k = 0; k < 6; ++k) {
        double pe = std::abs(last.pressure[k] - expected.pressure[k]),
               fe = std::abs(last.friction[k] - expected.friction[k]);
        pressure_error = std::max(pressure_error, pe);
        friction_error = std::max(friction_error, fe);
        if (pe > 2e-12 * (1 + std::abs(expected.pressure[k])) || fe > 1e-8 + 2e-6 * std::abs(expected.friction[k]) ||
            last.total[k] != last.pressure[k] + last.friction[k])
          return 11;
      }
      TransportLoads repeated;
      if (!gpu.EvaluateLoads(up, pp, repeated, density, viscosity) || !Same(last, repeated))
        return 12;
      for (size_t r = 0; r < devices.size(); ++r) {
        cudaSetDevice(devices[r]);
        for (bool cells : {false, true}) {
          auto &expected_values = cells ? local_p[r] : local_u[r];
          std::vector<double> actual(expected_values.size());
          if (cudaMemcpy(actual.data(), cells ? buffers.pressure[r] : buffers.velocity[r],
                         actual.size() * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess)
            return 13;
          for (size_t i = 0; i < actual.size(); ++i)
            if (actual[i] != expected_values[i] && !(std::isnan(actual[i]) && std::isnan(expected_values[i])))
              return 14;
        }
      }
      auto saved = last;
      if (gpu.EvaluateLoads(up, pp, last, 0, viscosity) || !Same(last, saved) ||
          gpu.EvaluateLoads(up, pp, last, density, -1.) || !Same(last, saved) || gpu.EvaluateLoads({}, pp, last) ||
          !Same(last, saved))
        return 15;
    }
  // Validate all owned entries, even inactive cells; reject before committing
  // loads if pressure on any rank is non-finite. Trailing NaNs above are ignored.
  auto saved = last;
  double bad = std::numeric_limits<double>::quiet_NaN();
  cudaSetDevice(devices.back());
  if (cudaMemcpy(buffers.pressure.back(), &bad, sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaDeviceSynchronize() != cudaSuccess)
    return 16;
  if (gpu.EvaluateLoads(up, pp, last) || gpu.Error().empty() || !Same(last, saved))
    return 17;
  double good = .3;
  if (cudaMemcpy(buffers.pressure.back(), &good, sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(buffers.velocity.back(), &bad, sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaDeviceSynchronize() != cudaSuccess)
    return 18;
  if (gpu.EvaluateLoads(up, pp, last) || !Same(last, saved))
    return 19;
  std::cout << "PASS distributed loads ranks=" << devices.size() << " mixed=" << mixed << " body=" << body
            << " width=" << width << " pressure_error=" << pressure_error << " friction_error=" << friction_error
            << '\n';
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
    for (int body : {0, 1, 2}) {
      if (mixed && body == 2)
        continue;
      for (double width : {1., 1.3})
        for (auto devices : mappings)
          if (int status = Run(mixed, body, width, devices)) {
            std::cerr << "load failure " << status << '\n';
            return status;
          }
    }
}
