#include "amr_gpu_transport.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

#include "amr_transport.h"

int Run(bool embedded, double width) {
  using namespace wind3d::amr;
  Topology tree;
  if (!tree.Regrid({2, 2, 2}, 1, 100, [](Block b) { return b.x == 0 && b.y == 0; }))
    return 1;
  Mesh mesh;
  if (!mesh.Build(tree, 4, width, {-width, -width, -width}))
    return 2;
  if (embedded && !mesh.Embed([width](std::array<double, 3> p) {
        return std::max({std::abs(p[0] + .5 * width), std::abs(p[1] + .5 * width), std::abs(p[2])}) - .21 * width;
      }))
    return 3;
  GpuTransport gpu, direct;
  if (!direct.Build(0, mesh, tree, width, {-width, -width, -width}, false))
    return 14;
  if (!gpu.Build(0, mesh, tree, width, {-width, -width, -width})) {
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
  FaceField cpu(mesh, tree, width, {-width, -width, -width});
  double max_error = 0, cache_error = 0;
  for (int pattern = 0; pattern < 3; ++pattern)
    for (bool cubic : {false, true})
      for (bool corrected : {false, true})
        for (double dt : {.07, 1.7, -.07}) {
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
              !(corrected ? cpu.AdvectCorrected(dt, reference, cubic) : cpu.Advect(dt, reference, cubic)))
            return 6;
          if (cudaMemcpy(device.u, input.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
              !gpu.AdvectDevice(device.u, device.u, nf, dt, cubic, corrected) ||
              cudaMemcpy(actual.data(), device.u, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
            std::cerr << gpu.Error() << '\n';
            return 7;
          }
          std::vector<double> direct_result(nf);
          if (cudaMemcpy(device.u, input.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
              !direct.AdvectDevice(device.u, device.u, nf, dt, cubic, corrected) ||
              cudaMemcpy(direct_result.data(), device.u, bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
            return 15;
          for (size_t i = 0; i < nf; ++i)
            cache_error = std::max(cache_error, std::abs(actual[i] - direct_result[i]));
          if (cache_error > 1e-12)
            return 16;
          double error = 0;
          for (size_t i = 0; i < nf; ++i)
            error = std::max(error, std::abs(actual[i] - reference[i]));
          max_error = std::max(max_error, error);
          if (error > 1e-9) {
            size_t worst = 0;
            for (size_t i = 0; i < nf; ++i)
              if (std::abs(actual[i] - reference[i]) > std::abs(actual[worst] - reference[worst]))
                worst = i;
            auto face = mesh.Faces()[worst];
            std::cerr << std::setprecision(17) << "width=" << width << " embedded=" << embedded
                      << " pattern=" << pattern << " cubic=" << cubic << " corrected=" << corrected << " dt=" << dt
                      << " error=" << error << " face=" << worst << " axis=" << face.axis
                      << " centre=" << face.centre[0] << ',' << face.centre[1] << ',' << face.centre[2]
                      << " CPU=" << reference[worst] << " GPU=" << actual[worst] << " direct=" << direct_result[worst]
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
  if (gpu.Build(0, empty, tree, width, {-width, -width, -width}) || gpu.Error().empty())
    return 11;
  if (!gpu.AdvectDevice(device.u, device.u, nf, .01))
    return 12;
  saved[0] = std::numeric_limits<double>::infinity();
  if (cudaMemcpy(device.u, saved.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
      gpu.AdvectDevice(device.u, device.u, nf, .01) ||
      cudaMemcpy(actual.data(), device.u, bytes, cudaMemcpyDeviceToHost) != cudaSuccess || actual != saved)
    return 13;
  std::cout << "PASS GPU sparse transport width=" << width << " embedded=" << embedded
            << " max CPU difference=" << max_error << " cache/direct difference=" << cache_error
            << " with aliasing and failure rollback\n";
  return 0;
}

int main() {
  for (double width : {1., 1.3, 2. / 3.})
    for (bool embedded : {false, true})
      if (int status = Run(embedded, width))
        return status;
  return 0;
}
