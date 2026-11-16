#include "amr_gpu_projection.h"

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <random>

#include "amr_mass.h"

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
    return 16;
  FaceGradient g;
  if (!g.Build(mesh))
    return 3;
  std::vector<double> u(mesh.Faces().size()), p;
  constexpr double pi = 3.141592653589793;
  for (size_t i = 0; i < u.size(); ++i) {
    auto f = mesh.Faces()[i];
    u[i] = std::sin((f.centre[f.axis] + 1) * pi * (f.axis == 0 ? .25 : 1));
  }
  auto before = u;
  auto reference_u = u;
  std::vector<double> reference_p;
  auto reference = ProjectReference(mesh, g, reference_u, reference_p);
  if (!reference.converged)
    return 8;
  GpuProjection gpu;
  if (!gpu.Build(0, mesh, g)) {
    std::cerr << gpu.Error() << '\n';
    return 9;
  }
  auto solve = gpu.Project(u, p);
  if (solve.converged) {
    double error = 0, scale = 0;
    for (size_t i = 0; i < u.size(); ++i) {
      error = std::max(error, std::abs(u[i] - reference_u[i]));
      scale = std::max(scale, std::abs(reference_u[i]));
    }
    std::cout << "CPU/GPU corrected velocity max difference=" << error << '\n';
    if (error > 1e-7 * std::max(1.0, scale))
      return 10;
  }
  GpuProjection scalar;
  auto scalar_u = before;
  std::vector<double> scalar_p;
  if (!scalar.Build(0, mesh, g, true, false) || !scalar.Project(scalar_u, scalar_p).converged)
    return 30;
  double row_difference = 0;
  for (size_t i = 0; i < u.size(); ++i)
    row_difference = std::max(row_difference, std::abs(u[i] - scalar_u[i]));
  for (size_t i = 0; i < p.size(); ++i)
    row_difference = std::max(row_difference, std::abs(p[i] - scalar_p[i]));
  if (!std::isfinite(row_difference) || row_difference > 1e-7)
    return 31;
  std::cout << "cooperative/scalar row maximum field difference=" << row_difference << '\n';
  GpuProjection diagonal;
  auto diagonal_u = before;
  std::vector<double> diagonal_p;
  if (!diagonal.Build(0, mesh, g, false))
    return 27;
  auto diagonal_result = diagonal.Project(diagonal_u, diagonal_p);
  if (!diagonal_result.converged || solve.iterations >= diagonal_result.iterations)
    return 28;
  double preconditioner_difference = 0;
  for (size_t i = 0; i < u.size(); ++i)
    preconditioner_difference = std::max(preconditioner_difference, std::abs(u[i] - diagonal_u[i]));
  if (preconditioner_difference > 1e-7)
    return 29;
  std::cout << "multilevel/Jacobi iterations=" << solve.iterations << '/' << diagonal_result.iterations
            << " field difference=" << preconditioner_difference << '\n';
  std::cout << "projection iterations=" << solve.iterations << " true relative residual=" << solve.relative_residual
            << '\n';
  if (!solve.converged)
    return 4;
  double initial = 0, final = 0;
  for (size_t i = 0; i < mesh.Cells().size(); ++i) {
    double a = 0, b = 0;
    for (int k = mesh.Offsets()[i]; k < mesh.Offsets()[i + 1]; ++k) {
      int id = mesh.Incidence()[k], f = std::abs(id) - 1;
      double area = (id > 0 ? 1 : -1) * mesh.Faces()[f].area * mesh.Faces()[f].aperture;
      a += area * before[f];
      b += area * u[f];
    }
    initial += a * a;
    final += b * b;
  }
  if (std::sqrt(final / initial) > 1.1e-9)
    return 5;
  for (size_t i = 0; i < u.size(); ++i) {
    auto f = mesh.Faces()[i];
    if ((f.low < 0 || f.high < 0) && !(f.axis == 0 && f.high < 0) && u[i] != before[i])
      return 6;
  }
  u = before;
  p.clear();
  auto fail = gpu.Project(u, p, 1e-12, 1);
  if (fail.converged || u != before || !p.empty())
    return 7;
  // Persistent workspace recovers after an iteration-limit failure.
  auto again = gpu.Project(u, p);
  if (!again.converged)
    return 11;
  std::vector<double> zero(u.size());
  p.clear();
  if (!gpu.Project(zero, p).converged)
    return 12;
  for (double value : p)
    if (value != 0)
      return 13;
  Mesh invalid;
  if (gpu.Build(0, invalid, g) || gpu.Error().empty())
    return 14;
  u = before;
  if (!gpu.Project(u, p).converged)
    return 15;

  // Exercise caller-owned resident buffers, including rejected/nonconverged calls.
  struct Buffers {
    double *u{}, *p{};

    ~Buffers() {
      cudaFree(u);
      cudaFree(p);
    }
  } device;

  const size_t ub = before.size() * sizeof(double), pb = mesh.Cells().size() * sizeof(double);
  if (cudaMalloc(&device.u, ub) != cudaSuccess || cudaMalloc(&device.p, pb) != cudaSuccess)
    return 17;
  std::vector<double> sentinel(mesh.Cells().size(), 42), check_u(before.size()), check_p(sentinel.size());
  auto upload = [&] {
    return cudaMemcpy(device.u, before.data(), ub, cudaMemcpyHostToDevice) == cudaSuccess &&
           cudaMemcpy(device.p, sentinel.data(), pb, cudaMemcpyHostToDevice) == cudaSuccess;
  };
  auto download = [&] {
    return cudaMemcpy(check_u.data(), device.u, ub, cudaMemcpyDeviceToHost) == cudaSuccess &&
           cudaMemcpy(check_p.data(), device.p, pb, cudaMemcpyDeviceToHost) == cudaSuccess;
  };
  if (!upload())
    return 18;
  if (gpu.ProjectDevice(device.u, before.size(), device.p, sentinel.size(), 1e-12, 1).converged || !download() ||
      check_u != before || check_p != sentinel)
    return 19;
  if (gpu.ProjectDevice(device.u, before.size() - 1, device.p, sentinel.size()).converged || !download() ||
      check_u != before || check_p != sentinel)
    return 20;
  if (!gpu.ProjectDevice(device.u, before.size(), device.p, sentinel.size()).converged || !download())
    return 21;
  for (size_t f = 0; f < check_u.size(); ++f)
    if (std::abs(check_u[f] - reference_u[f]) > 1e-7)
      return 22;
  for (size_t c = 0; c < check_p.size(); ++c)
    if (std::abs(check_p[c] - reference_p[c]) > 1e-7)
      return 23;
  // Zero divergence must still clear solid faces and return zero potential.
  for (size_t f = 0; f < before.size(); ++f)
    before[f] = mesh.Faces()[f].aperture == 0 ? 17 : 0;
  if (!upload() || !gpu.ProjectDevice(device.u, before.size(), device.p, sentinel.size()).converged || !download())
    return 24;
  for (double v : check_u)
    if (v != 0)
      return 25;
  for (double v : check_p)
    if (v != 0)
      return 26;
  std::cout << "PASS resident GPU buffers, CPU agreement, failure rollback and closed-face clearing\n";
  std::cout
      << "PASS composite projection removes divergence, preserves fixed boundary flux and rolls back nonconvergence\n";
  return 0;
}

bool Compatible(bool mixed, bool embedded, double width) {
  using namespace wind3d::amr;
  Topology tree;
  if (!tree.Regrid({2, 2, 2}, 1, 100, [mixed](Block b) { return mixed && b.x == 0; }))
    return false;
  Mesh mesh;
  if (!mesh.Build(tree, 4, width))
    return false;
  if (embedded && !mesh.Embed([width](std::array<double, 3> p) {
        return std::max({std::abs(p[0] - .5 * width), std::abs(p[1] - .5 * width), std::abs(p[2] - .5 * width)}) -
               .21 * width;
      }))
    return false;
  FaceGradient g;
  FaceMass mass;
  GpuProjection gpu;
  if (!g.Build(mesh) || !mass.Build(mesh) || !gpu.Build(0, mesh, g, true, true, true)) {
    std::cerr << gpu.Error() << '\n';
    return false;
  }
  std::mt19937 random(123);
  std::uniform_real_distribution<double> distribution(-1, 1);
  std::vector<double> u(mesh.Faces().size()), p;
  for (size_t i = 0; i < u.size(); ++i) {
    auto f = mesh.Faces()[i];
    if (f.aperture > 0 && ((f.low >= 0 && f.high >= 0) || (f.axis == 0 && f.high < 0)))
      u[i] = distribution(random);
  }
  auto before = u;
  auto result = gpu.Project(u, p, 1e-11);
  if (!result.converged) {
    std::cerr << "Compatible failed " << gpu.Error() << '\n';
    return false;
  }
  std::vector<double> correction(u.size());
  for (size_t i = 0; i < u.size(); ++i)
    correction[i] = before[i] - u[i];
  auto product = [&](const std::vector<double> &a, const std::vector<double> &b) {
    double sum = 0;
    for (size_t i = 0; i < a.size(); ++i)
      for (int k = mass.Offsets()[i]; k < mass.Offsets()[i + 1]; ++k)
        sum += a[i] * mass.Weights()[k] * b[mass.Columns()[k]];
    return sum;
  };
  double initial = product(before, before), final = product(u, u), removed = product(correction, correction);
  double balance = std::abs(initial - final - removed) / initial;
  if (final > initial * (1 + 1e-10) || balance > 1e-9)
    return false;
  double div0 = 0, div1 = 0;
  for (size_t c = 0; c < mesh.Cells().size(); ++c) {
    double a = 0, b = 0;
    for (int k = mesh.Offsets()[c]; k < mesh.Offsets()[c + 1]; ++k) {
      int signed_face = mesh.Incidence()[k], f = std::abs(signed_face) - 1;
      double weight = (signed_face > 0 ? 1 : -1) * mesh.Faces()[f].area * mesh.Faces()[f].aperture;
      a += weight * before[f];
      b += weight * u[f];
    }
    div0 += a * a;
    div1 += b * b;
  }
  if (std::sqrt(div1 / div0) > 2e-11)
    return false;
  GpuProjection bicg, jacobi;
  auto bicg_u = before, jacobi_u = before;
  std::vector<double> bicg_p, jacobi_p;
  if (!bicg.Build(0, mesh, g, true, true, true, false) || !bicg.Project(bicg_u, bicg_p, 1e-11).converged ||
      !jacobi.Build(0, mesh, g, false, true, true) || !jacobi.Project(jacobi_u, jacobi_p, 1e-11).converged)
    return false;
  for (size_t i = 0; i < u.size(); ++i)
    if (std::abs(u[i] - bicg_u[i]) > 1e-8 || std::abs(u[i] - jacobi_u[i]) > 1e-8)
      return false;
  auto once = u;
  if (!gpu.Project(u, p, 1e-9).converged)
    return false;
  for (size_t i = 0; i < u.size(); ++i)
    if (std::abs(u[i] - once[i]) > 1e-8)
      return false;
  if (!mixed) {
    auto reference = before;
    std::vector<double> reference_p;
    if (!ProjectReference(mesh, g, reference, reference_p, 1e-11).converged)
      return false;
    for (size_t i = 0; i < u.size(); ++i)
      if (std::abs(u[i] - reference[i]) > 1e-8)
        return false;
  }
  auto failed = before;
  std::vector<double> sentinel(3, 17);
  if (gpu.Project(failed, sentinel, 1e-14, 1).converged || failed != before || sentinel != std::vector<double>(3, 17))
    return false;
  std::cout << "Compatible mixed=" << mixed << " embedded=" << embedded << " width=" << width
            << " iterations=" << result.iterations << " divergence=" << std::sqrt(div1 / div0)
            << " energy ratio=" << final / initial << " balance=" << balance << '\n';
  return true;
}

int main() {
  for (bool mixed : {false, true})
    for (bool embedded : {false, true})
      for (double width : {1., 1.3})
        if (!Compatible(mixed, embedded, width))
          return 40;
  int status = Run(false);
  return status ? status : Run(true);
}
