#include "amr_gpu_transport.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

#include "amr_transport.h"

int Run(bool embedded) {
  using namespace wind3d::amr;
  Topology tree;
  if (!tree.Regrid({2, 2, 2}, 1, 100, [](Block b) { return b.x == 0 && b.y == 0; }))
    return 1;
  Mesh mesh;
  if (!mesh.Build(tree, 4, 1, {-1, -1, -1}))
    return 2;
  if (embedded && !mesh.Embed([](std::array<double, 3> p) {
        return std::max({std::abs(p[0] + .5), std::abs(p[1] + .5), std::abs(p[2])}) - .21;
      }))
    return 3;
  GpuTransport gpu;
  if (!gpu.Build(0, mesh, tree, 1, {-1, -1, -1})) {
    std::cerr << gpu.Error() << '\n';
    return 4;
  }

  struct Buffer {
    double *u{};

    ~Buffer() {
      cudaFree(u);
    }
  } device;

  size_t nf = mesh.Faces().size(), bytes = nf * sizeof(double);
  if (cudaMalloc(&device.u, bytes) != cudaSuccess)
    return 5;
  FaceField cpu(mesh, tree, 1, {-1, -1, -1});
  double max_error = 0;
  for (int pattern = 0; pattern < 3; ++pattern)
    for (bool cubic : {false, true})
      for (bool corrected : {false, true}) {
        std::vector<double> input(nf), reference, actual(nf);
        for (size_t i = 0; i < nf; ++i) {
          auto f = mesh.Faces()[i];
          input[i] = pattern == 0   ? .3
                     : pattern == 1 ? .2 + .1 * std::sin(2 * f.centre[(f.axis + 1) % 3])
                                    : (f.centre[(f.axis + 1) % 3] > 0 ? .3 : .1);
          if (f.aperture == 0)
            input[i] = 0;
        }
        auto extended = input;
        if (!cpu.ExtendSlip(extended) || !cpu.Update(extended) ||
            !(corrected ? cpu.AdvectCorrected(.07, reference, cubic) : cpu.Advect(.07, reference, cubic)))
          return 6;
        if (cudaMemcpy(device.u, input.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
            !gpu.AdvectDevice(device.u, device.u, nf, .07, cubic, corrected) ||
            cudaMemcpy(actual.data(), device.u, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
          std::cerr << gpu.Error() << '\n';
          return 7;
        }
        double error = 0;
        for (size_t i = 0; i < nf; ++i)
          error = std::max(error, std::abs(actual[i] - reference[i]));
        max_error = std::max(max_error, error);
        if (error > 1e-9) {
          std::cerr << "pattern=" << pattern << " cubic=" << cubic << " corrected=" << corrected << " error=" << error
                    << '\n';
          return 8;
        }
      }
  std::vector<double> saved(nf, .25), actual(nf);
  if (cudaMemcpy(device.u, saved.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess)
    return 9;
  if (gpu.AdvectDevice(device.u, device.u, nf, std::numeric_limits<double>::infinity()) ||
      gpu.AdvectDevice(device.u, device.u, nf - 1, .01) ||
      cudaMemcpy(actual.data(), device.u, bytes, cudaMemcpyDeviceToHost) != cudaSuccess || actual != saved)
    return 10;
  Mesh empty;
  if (gpu.Build(0, empty, tree, 1, {-1, -1, -1}) || gpu.Error().empty())
    return 11;
  if (!gpu.AdvectDevice(device.u, device.u, nf, .01))
    return 12;
  saved[0] = std::numeric_limits<double>::infinity();
  if (cudaMemcpy(device.u, saved.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
      gpu.AdvectDevice(device.u, device.u, nf, .01) ||
      cudaMemcpy(actual.data(), device.u, bytes, cudaMemcpyDeviceToHost) != cudaSuccess || actual != saved)
    return 13;
  std::cout << "PASS GPU sparse transport embedded=" << embedded << " max CPU difference=" << max_error
            << " with aliasing and failure rollback\n";
  return 0;
}

int main() {
  int status = Run(false);
  return status ? status : Run(true);
}
