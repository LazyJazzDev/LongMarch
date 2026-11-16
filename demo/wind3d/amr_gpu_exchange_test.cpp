#include "amr_gpu_exchange.h"

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <limits>

namespace {
struct Buffers {
  std::vector<int> devices;
  std::vector<double *> cells, faces, snapshots;
  std::vector<cudaStream_t> streams;

  ~Buffers() {
    int previous = 0;
    cudaGetDevice(&previous);
    for (size_t r = 0; r < devices.size(); ++r) {
      cudaSetDevice(devices[r]);
      if (streams[r]) {
        cudaStreamSynchronize(streams[r]);
        cudaStreamDestroy(streams[r]);
      }
      cudaFree(cells[r]);
      cudaFree(faces[r]);
      cudaFree(snapshots[r]);
    }
    cudaSetDevice(previous);
  }
};
}  // namespace

int Run(bool mixed, std::vector<int> devices, int selection) {
  using namespace wind3d::amr;
  Topology tree;
  Mesh mesh;
  FaceMass mass;
  PressurePartition plan;
  if (!tree.Regrid({2, 2, 2}, 1, 64, [mixed](Block b) { return !mixed || (b.x == 0 && b.y == 0); }) ||
      !mesh.Build(tree, 4, 1.3, {-1.3, -1.3, -1.3}) || !mass.Build(mesh) ||
      !plan.Build(mesh, tree, mass, int(devices.size())))
    return 1;
  std::vector<unsigned char> mask;
  if (selection) {
    mask.resize(mesh.Faces().size());
    for (size_t i = 0; i < mask.size(); ++i)
      mask[i] = (selection == 2 && i % 3 == 1) || (selection == 3 && plan.FaceOwners()[i] == 0);
  }
  GpuPressureExchange exchange;
  int previous = devices.back();
  if (cudaSetDevice(previous) != cudaSuccess || !exchange.Build(plan, devices, mask)) {
    std::cerr << exchange.Error() << '\n';
    return 2;
  }
  int current = -1;
  cudaGetDevice(&current);
  if (current != previous)
    return 3;
  const auto &layouts = exchange.Layouts();
  if (exchange.Build(plan, devices, std::vector<unsigned char>(1, 1)))
    return 15;
  GpuPressureExchange full;
  if (!full.Build(plan, devices))
    return 16;
  for (size_t r = 0; r < layouts.size(); ++r)
    if (layouts[r].cells != full.Layouts()[r].cells || layouts[r].faces != full.Layouts()[r].faces ||
        layouts[r].owned_cells != full.Layouts()[r].owned_cells ||
        layouts[r].owned_faces != full.Layouts()[r].owned_faces)
      return 17;
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
  values.snapshots.resize(devices.size());
  values.streams.resize(devices.size());
  for (size_t r = 0; r < devices.size(); ++r) {
    if (cudaSetDevice(devices[r]) != cudaSuccess ||
        cudaMalloc(&values.cells[r], layouts[r].cells.size() * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&values.faces[r], layouts[r].faces.size() * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&values.snapshots[r], layouts[r].faces.size() * sizeof(double)) != cudaSuccess ||
        cudaStreamCreateWithFlags(&values.streams[r], cudaStreamNonBlocking) != cudaSuccess)
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
            cudaMemcpy(buffers[r], host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaDeviceSynchronize() != cudaSuccess)
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
  // Repeated queued imports keep sparse destination slots, with all producers
  // and consumers ordered on the caller streams. The full synchronous path
  // above must still populate every ghost despite the queued-only selection.
  for (int repeat = 0; repeat < 8; ++repeat) {
    for (size_t r = 0; r < devices.size(); ++r) {
      std::vector<double> host(layouts[r].faces.size(), -12345.);
      for (size_t i = 0; i < layouts[r].owned_faces; ++i)
        host[i] = layouts[r].faces[i] * 7. + repeat;
      if (cudaSetDevice(devices[r]) != cudaSuccess ||
          cudaMemcpy(values.faces[r], host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice) !=
              cudaSuccess ||
          cudaDeviceSynchronize() != cudaSuccess)
        return 18;
    }
    if (cudaSetDevice(previous) != cudaSuccess || !exchange.EnqueueFaces(values.faces, values.streams) ||
        !exchange.EnqueueFaces(values.faces, values.streams))
      return 19;
    cudaGetDevice(&current);
    if (current != previous)
      return 20;
    // Snapshot ghost consumers, then change owned producers on each stream.
    // The next exchange must preserve the snapshot and publish the new values
    // without a host barrier between these stages, including one-way links.
    for (size_t r = 0; r < devices.size(); ++r)
      if (cudaSetDevice(devices[r]) != cudaSuccess ||
          cudaMemcpyAsync(values.snapshots[r], values.faces[r], layouts[r].faces.size() * sizeof(double),
                          cudaMemcpyDeviceToDevice, values.streams[r]) != cudaSuccess ||
          cudaMemsetAsync(values.faces[r], 0, layouts[r].owned_faces * sizeof(double), values.streams[r]) !=
              cudaSuccess)
        return 23;
    if (!exchange.EnqueueFaces(values.faces, values.streams))
      return 24;
    for (size_t r = 0; r < devices.size(); ++r) {
      std::vector<double> actual(layouts[r].faces.size());
      if (cudaSetDevice(devices[r]) != cudaSuccess || cudaStreamSynchronize(values.streams[r]) != cudaSuccess ||
          cudaMemcpy(actual.data(), values.snapshots[r], actual.size() * sizeof(double), cudaMemcpyDeviceToHost) !=
              cudaSuccess)
        return 21;
      for (size_t i = 0; i < actual.size(); ++i) {
        int id = layouts[r].faces[i];
        double expected = i < layouts[r].owned_faces || mask.empty() || mask[id] ? id * 7. + repeat : -12345.;
        if (actual[i] != expected)
          return 22;
      }
      if (cudaMemcpy(actual.data(), values.faces[r], actual.size() * sizeof(double), cudaMemcpyDeviceToHost) !=
          cudaSuccess)
        return 25;
      for (size_t i = 0; i < actual.size(); ++i) {
        int id = layouts[r].faces[i];
        double expected = i < layouts[r].owned_faces || mask.empty() || mask[id] ? 0. : -12345.;
        if (actual[i] != expected)
          return 26;
      }
    }
  }
  std::cout << "PASS peer exchange mixed=" << mixed << " selection=" << selection << " ranks=" << devices.size()
            << '\n';
  return 0;
}

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2)
    return 77;
  std::vector<std::vector<int>> mappings{{0}, {0, 1}, {1, 0}};
  if (devices >= 4) {
    mappings.push_back({0, 1, 2, 3});
    mappings.push_back({3, 2, 1, 0});
  }
  for (bool mixed : {false, true})
    for (auto mapping : mappings)
      for (int selection : {0, 1, 2, 3})
        if (int status = Run(mixed, mapping, selection)) {
          std::cerr << "exchange failure " << status << '\n';
          return status;
        }
}
