#include "amr_gpu_distributed_transport.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

#include "amr_gpu_transport.h"
#include "amr_partition.h"

namespace {
struct Buffers {
  std::vector<int> devices;
  std::vector<double *> u, v;
  double *reference{};

  ~Buffers() {
    int previous;
    cudaGetDevice(&previous);
    for (size_t r = 0; r < devices.size(); ++r) {
      cudaSetDevice(devices[r]);
      cudaFree(u[r]);
      cudaFree(v[r]);
    }
    if (!devices.empty()) {
      cudaSetDevice(devices[0]);
      cudaFree(reference);
    }
    cudaSetDevice(previous);
  }
};

int Run(bool mixed, bool body, double width, const std::vector<int> &devices) {
  using namespace wind3d::amr;
  Topology tree;
  Mesh mesh;
  FaceMass mass;
  PressurePartition partition;
  if (!tree.Regrid({2, 2, 2}, 1, 100, [mixed](Block b) { return !mixed || (b.x == 0 && b.y == 0); }) ||
      !mesh.Build(tree, 4, width, {-width, -width, -width}))
    return 1;
  if (body && !mesh.Embed([width](std::array<double, 3> p) {
        return std::max({std::abs(p[0] + .5 * width), std::abs(p[1] + .5 * width), std::abs(p[2])}) - .21 * width;
      }))
    return 2;
  if (!mass.Build(mesh) || !partition.Build(mesh, tree, mass, int(devices.size())))
    return 3;
  GpuDistributedTransport distributed;
  GpuTransport reference;
  int previous = devices.back(), current = -1;
  cudaSetDevice(previous);
  if (!distributed.Build(mesh, tree, devices, width, {-width, -width, -width})) {
    std::cerr << distributed.Error() << '\n';
    return 4;
  }
  cudaGetDevice(&current);
  if (current != previous || distributed.OwnedFaces() != partition.OwnedFaces())
    return 5;
  if (distributed.Build(mesh, tree, {}, width, {}) || distributed.Error().empty() ||
      distributed.Build(mesh, tree, {-1}, width, {}) ||
      distributed.Build(mesh, tree, {devices[0], devices[0]}, width, {}) ||
      distributed.OwnedFaces() != partition.OwnedFaces())
    return 6;
  if (!reference.Build(devices[0], mesh, tree, width, {-width, -width, -width}, false))
    return 7;
  Buffers buffers;
  buffers.devices = devices;
  buffers.u.resize(devices.size());
  buffers.v.resize(devices.size());
  cudaSetDevice(devices[0]);
  if (cudaMalloc(&buffers.reference, mesh.Faces().size() * sizeof(double)) != cudaSuccess)
    return 8;
  for (size_t r = 0; r < devices.size(); ++r) {
    cudaSetDevice(devices[r]);
    size_t n = distributed.OwnedFaces()[r].size() + 5;
    if (cudaMalloc(&buffers.u[r], n * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&buffers.v[r], n * sizeof(double)) != cudaSuccess)
      return 9;
  }
  double maximum = 0;
  int repeat = 0;
  for (double dt : {0., .07, 1.7, -.07})
    for (bool cubic : {false, true})
      for (bool corrected : {false, true}) {
        ++repeat;
        std::optional<double> inlet =
            repeat % 3 == 0 ? std::optional<double>{} : std::optional<double>{repeat % 3 == 1 ? .7 : -.2};
        std::vector<double> input(mesh.Faces().size()), expected(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
          auto f = mesh.Faces()[i];
          input[i] = repeat % 2 ? .3 * std::sin(2 * f.centre[(f.axis + 1) % 3]) + .1 * f.axis
                                : (f.centre[(f.axis + 1) % 3] > 0 ? .3 : -.1);
        }
        cudaSetDevice(devices[0]);
        if (cudaMemcpy(buffers.reference, input.data(), input.size() * sizeof(double), cudaMemcpyHostToDevice) !=
                cudaSuccess ||
            cudaDeviceSynchronize() != cudaSuccess ||
            !reference.AdvectDevice(buffers.reference, buffers.reference, input.size(), dt, cubic, corrected, inlet) ||
            cudaMemcpy(expected.data(), buffers.reference, expected.size() * sizeof(double), cudaMemcpyDeviceToHost) !=
                cudaSuccess)
          return 10;
        std::vector<const double *> inputs;
        auto outputs = repeat % 2 ? buffers.u : buffers.v;
        for (size_t r = 0; r < devices.size(); ++r) {
          auto &ids = distributed.OwnedFaces()[r];
          std::vector<double> local(ids.size() + 5, -9876.), sentinel(local.size(), -9876.);
          for (size_t i = 0; i < ids.size(); ++i)
            local[i] = input[ids[i]];
          cudaSetDevice(devices[r]);
          if (cudaMemcpy(buffers.u[r], local.data(), local.size() * sizeof(double), cudaMemcpyHostToDevice) !=
                  cudaSuccess ||
              cudaMemcpy(buffers.v[r], sentinel.data(), sentinel.size() * sizeof(double), cudaMemcpyHostToDevice) !=
                  cudaSuccess ||
              cudaDeviceSynchronize() != cudaSuccess)
            return 11;
          inputs.push_back(buffers.u[r]);
        }
        cudaSetDevice(previous);
        if (!distributed.Advect(inputs, outputs, dt, cubic, corrected, inlet)) {
          std::cerr << distributed.Error() << '\n';
          return 12;
        }
        cudaGetDevice(&current);
        if (current != previous)
          return 13;
        for (size_t r = 0; r < devices.size(); ++r) {
          auto &ids = distributed.OwnedFaces()[r];
          std::vector<double> actual(ids.size() + 5);
          cudaSetDevice(devices[r]);
          if (cudaMemcpy(actual.data(), outputs[r], actual.size() * sizeof(double), cudaMemcpyDeviceToHost) !=
              cudaSuccess)
            return 14;
          for (size_t i = 0; i < ids.size(); ++i) {
            if (!std::isfinite(actual[i]))
              return 15;
            maximum = std::max(maximum, std::abs(actual[i] - expected[ids[i]]));
          }
          for (size_t i = ids.size(); i < actual.size(); ++i)
            if (actual[i] != -9876.)
              return 16;
        }
        if (maximum > 2e-11) {
          std::cerr << "difference=" << maximum << '\n';
          return 17;
        }
      }
  double diffusion_error = 0, min_area = std::numeric_limits<double>::infinity();
  for (auto f : mesh.Faces())
    min_area = std::min(min_area, f.area);
  for (bool cubic : {false, true})
    for (bool damped : {false, true}) {
      double dt = .03, speed = damped ? -.4 : .4;
      double viscosity = damped ? .1 * min_area / dt : 0;
      double sponge = damped ? .75 * width : 0;
      std::vector<double> input(mesh.Faces().size()), expected;
      for (size_t i = 0; i < input.size(); ++i) {
        auto f = mesh.Faces()[i];
        input[i] = .3 * std::sin(2 * f.centre[(f.axis + 1) % 3]) + .1 * f.axis;
      }
      FaceField cpu(mesh, tree, width, {-width, -width, -width}, speed);
      auto extended = input;
      if (!cpu.ExtendSlip(extended) || !cpu.Update(extended) || !cpu.AdvectCorrected(dt, expected, cubic) ||
          !cpu.DiffuseAndSponge(dt, viscosity, sponge, speed, expected))
        return 24;
      for (size_t i = 0; i < expected.size(); ++i) {
        auto f = mesh.Faces()[i];
        if (f.aperture == 0)
          expected[i] = 0;
        else if ((f.low < 0 || f.high < 0) && !(f.axis == 0 && f.high < 0))
          expected[i] = f.axis == 0 ? speed : 0;
      }
      std::vector<const double *> inputs;
      auto outputs = cubic ? buffers.u : buffers.v;
      for (size_t r = 0; r < devices.size(); ++r) {
        auto &ids = distributed.OwnedFaces()[r];
        std::vector<double> local(ids.size() + 5, -9876.), sentinel(local.size(), -9876.);
        for (size_t i = 0; i < ids.size(); ++i)
          local[i] = input[ids[i]];
        cudaSetDevice(devices[r]);
        if (cudaMemcpy(buffers.u[r], local.data(), local.size() * sizeof(double), cudaMemcpyHostToDevice) !=
                cudaSuccess ||
            cudaMemcpy(buffers.v[r], sentinel.data(), sentinel.size() * sizeof(double), cudaMemcpyHostToDevice) !=
                cudaSuccess ||
            cudaDeviceSynchronize() != cudaSuccess)
          return 25;
        inputs.push_back(buffers.u[r]);
      }
      cudaSetDevice(previous);
      if (!distributed.AdvectDiffuse(inputs, outputs, dt, speed, cubic, viscosity, sponge)) {
        std::cerr << distributed.Error() << '\n';
        return 26;
      }
      cudaGetDevice(&current);
      if (current != previous)
        return 27;
      for (size_t r = 0; r < devices.size(); ++r) {
        auto &ids = distributed.OwnedFaces()[r];
        std::vector<double> actual(ids.size() + 5);
        cudaSetDevice(devices[r]);
        if (cudaMemcpy(actual.data(), outputs[r], actual.size() * sizeof(double), cudaMemcpyDeviceToHost) !=
            cudaSuccess)
          return 28;
        for (size_t i = 0; i < ids.size(); ++i) {
          if (!std::isfinite(actual[i]))
            return 29;
          diffusion_error = std::max(diffusion_error, std::abs(actual[i] - expected[ids[i]]));
        }
        for (size_t i = ids.size(); i < actual.size(); ++i)
          if (actual[i] != -9876.)
            return 30;
        std::fill(actual.begin(), actual.end(), -9876.);
        if (cudaMemcpy(buffers.v[r], actual.data(), actual.size() * sizeof(double), cudaMemcpyHostToDevice) !=
                cudaSuccess ||
            cudaDeviceSynchronize() != cudaSuccess)
          return 31;
      }
      if (diffusion_error > 2e-11) {
        std::cerr << "diffusion difference=" << diffusion_error << '\n';
        return 32;
      }
      if (distributed.AdvectDiffuse(inputs, buffers.v, dt, speed, cubic, min_area / dt, sponge) ||
          distributed.AdvectDiffuse(inputs, buffers.v, dt, speed, cubic, -1., sponge) ||
          distributed.AdvectDiffuse(inputs, buffers.v, dt, speed, cubic, 0, -1.) ||
          distributed.AdvectDiffuse(inputs, buffers.v, 0, speed, cubic, 0, 0) ||
          distributed.AdvectDiffuse(inputs, buffers.v, dt, std::numeric_limits<double>::infinity(), cubic, 0, 0))
        return 33;
      for (size_t r = 0; r < devices.size(); ++r) {
        std::vector<double> actual(distributed.OwnedFaces()[r].size() + 5);
        cudaSetDevice(devices[r]);
        if (cudaMemcpy(actual.data(), buffers.v[r], actual.size() * sizeof(double), cudaMemcpyDeviceToHost) !=
            cudaSuccess)
          return 34;
        for (double value : actual)
          if (value != -9876.)
            return 35;
      }
    }
  // A non-finite value on any rank must prevent commits on every rank.
  std::vector<const double *> inputs;
  for (size_t r = 0; r < devices.size(); ++r) {
    std::vector<double> local(distributed.OwnedFaces()[r].size() + 5, -9876.);
    cudaSetDevice(devices[r]);
    if (cudaMemcpy(buffers.v[r], local.data(), local.size() * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess)
      return 18;
    if (r + 1 == devices.size()) {
      double bad = std::numeric_limits<double>::quiet_NaN();
      if (cudaMemcpy(buffers.u[r], &bad, sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess)
        return 19;
    }
    if (cudaDeviceSynchronize() != cudaSuccess)
      return 20;
    inputs.push_back(buffers.u[r]);
  }
  if (distributed.Advect(inputs, buffers.v, .1) || distributed.Error().empty() ||
      distributed.AdvectDiffuse(inputs, buffers.v, .1, .3) || distributed.Error().empty())
    return 21;
  for (size_t r = 0; r < devices.size(); ++r) {
    std::vector<double> actual(distributed.OwnedFaces()[r].size() + 5);
    cudaSetDevice(devices[r]);
    if (cudaMemcpy(actual.data(), buffers.v[r], actual.size() * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess)
      return 22;
    for (double value : actual)
      if (value != -9876.)
        return 23;
  }
  std::cout << "PASS distributed transport ranks=" << devices.size() << " mixed=" << mixed << " body=" << body
            << " width=" << width << " maximum=" << maximum << " diffusion_cpu_error=" << diffusion_error << '\n';
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
            std::cerr << "transport failure " << status << '\n';
            return status;
          }
}
