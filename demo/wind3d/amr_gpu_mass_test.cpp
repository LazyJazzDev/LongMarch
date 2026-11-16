#include "amr_gpu_mass.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

#include "amr_mass.h"

bool Run(int mask, double width, bool embedded, bool boundaries) {
  using namespace wind3d::amr;
  Topology tree;
  if (!tree.Regrid({3, 3, 3}, 1, 256, [mask](Block b) {
        int d[3]{b.x - 1, b.y - 1, b.z - 1};
        if (std::abs(d[0]) + std::abs(d[1]) + std::abs(d[2]) != 1)
          return false;
        int axis = d[0] ? 0 : (d[1] ? 1 : 2);
        return bool(mask & (1 << (2 * axis + (d[axis] > 0))));
      }))
    return false;
  Mesh mesh;
  if (!mesh.Build(tree, 4, width))
    return false;
  if (embedded && !mesh.Embed([width](std::array<double, 3> p) {
        return std::max({std::abs(p[0] - .5 * width), std::abs(p[1] - 1.5 * width), std::abs(p[2] - 1.5 * width)}) -
               .2 * width;
      }))
    return false;
  FaceMass mass;
  GpuMassInverse inverse;
  if (!mass.Build(mesh) || !inverse.Build(0, mesh, boundaries)) {
    std::cerr << inverse.Error() << '\n';
    return false;
  }
  size_t n = mesh.Faces().size(), bytes = n * sizeof(double);

  struct Buffers {
    double *rhs{}, *result{};

    ~Buffers() {
      cudaFree(rhs);
      cudaFree(result);
    }
  } device;

  if (cudaMalloc(&device.rhs, bytes) != cudaSuccess || cudaMalloc(&device.result, bytes) != cudaSuccess)
    return false;
  std::vector<unsigned char> free(n);
  std::vector<double> a(n), b(n), sum(n), x(n), y(n), z(n);
  std::mt19937 random(123);
  std::uniform_real_distribution<double> distribution(-1, 1);
  for (size_t i = 0; i < n; ++i) {
    auto f = mesh.Faces()[i];
    free[i] = f.aperture > 0 && (!boundaries || (f.low >= 0 && f.high >= 0) || (f.axis == 0 && f.high < 0));
    a[i] = distribution(random);
    b[i] = distribution(random);
    sum[i] = a[i] + b[i];
  }
  auto apply = [&](const std::vector<double> &rhs, std::vector<double> &out) {
    return cudaMemcpy(device.rhs, rhs.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
           inverse.ApplyDevice(device.rhs, device.result, n) && cudaDeviceSynchronize() == cudaSuccess &&
           cudaMemcpy(out.data(), device.result, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
  };
  if (!apply(a, x) || !apply(b, y) || !apply(sum, z))
    return false;
  double residual = 0, norm = 0, linearity = 0, scale = 0, ab = 0, ba = 0, energy = 0;
  for (size_t i = 0; i < n; ++i) {
    if (!free[i]) {
      if (x[i] != 0 || y[i] != 0 || z[i] != 0)
        return false;
    } else {
      double product = 0;
      for (int k = mass.Offsets()[i]; k < mass.Offsets()[i + 1]; ++k)
        product += mass.Weights()[k] * x[mass.Columns()[k]];
      residual += (product - a[i]) * (product - a[i]);
      norm += a[i] * a[i];
    }
    linearity = std::max(linearity, std::abs(z[i] - x[i] - y[i]));
    scale = std::max(scale, std::max(std::abs(x[i]), std::abs(y[i])));
    ab += a[i] * y[i];
    ba += b[i] * x[i];
    energy += a[i] * x[i];
  }
  if (!(energy > 0) || std::sqrt(residual / norm) > 5e-12 || linearity > 5e-12 * scale ||
      std::abs(ab - ba) > 5e-12 * energy)
    return false;
  // Rejected arguments/builds must not overwrite outputs or destroy the old operator.
  if (inverse.ApplyDevice(device.rhs, device.result, n - 1) || inverse.ApplyDevice(device.rhs, device.rhs, n) ||
      inverse.Build(0, Mesh{}, boundaries))
    return false;
  if (cudaMemcpy(z.data(), device.result, bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
    return false;
  auto preserved = z;
  if (!apply(sum, z) || z != preserved)
    return false;
  if (!mask && inverse.CoupledFaces() != 0)
    return false;
  std::cout << "mass inverse mask=" << mask << " width=" << width << " embedded=" << embedded
            << " boundaries=" << boundaries << " coupled=" << inverse.CoupledFaces()
            << " residual=" << std::sqrt(residual / norm) << '\n';
  return true;
}

int main() {
  for (int mask = 0; mask < 64; ++mask)
    for (bool boundaries : {false, true})
      if (!Run(mask, 1, false, boundaries))
        return 1;
  for (int mask : {1, 21, 63})
    for (bool embedded : {false, true})
      for (bool boundaries : {false, true})
        if (!Run(mask, 1.3, embedded, boundaries))
          return 2;
  std::cout << "PASS fixed polynomial mass inverse, true residual, linearity, symmetry, boundaries and rollback\n";
}
