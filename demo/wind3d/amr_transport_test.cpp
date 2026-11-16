#include "amr_transport.h"

#include <cmath>
#include <iostream>
#include <random>

int Run(double width) {
  using namespace wind3d::amr;
  Topology tree;
  if (!tree.Regrid({2, 2, 2}, 1, 100, [](Block b) { return b.x == 0 && b.y == 0; }))
    return 1;
  Mesh mesh;
  if (!mesh.Build(tree, 4, width, {-width, -width, -width}))
    return 2;
  FaceField field(mesh, tree, width, {-width, -width, -width});
  std::vector<double> u(mesh.Faces().size());
  for (size_t i = 0; i < u.size(); ++i)
    u[i] = mesh.Faces()[i].axis == 0 ? .7 : mesh.Faces()[i].axis == 1 ? .2 : -.1;
  if (!field.Update(u))
    return 3;
  std::vector<double> moved;
  if (!field.Advect(.01 * width, moved))
    return 4;
  for (size_t i = 0; i < u.size(); ++i)
    if (std::abs(u[i] - moved[i]) > 1e-12)
      return 5;
  for (size_t i = 0; i < u.size(); ++i) {
    auto f = mesh.Faces()[i];
    u[i] = f.axis == 0 ? .3 + f.centre[1] : f.axis == 1 ? .2 : .1;
  }
  if (!field.Update(u))
    return 6;
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> random(-.6 * width, .6 * width);
  double error = 0;
  for (int i = 0; i < 1000; ++i) {
    std::array<double, 3> p{random(rng), random(rng), random(rng)};
    auto v = field.Sample(p);
    for (int axis = 0; axis < 3; ++axis)
      if (field.SampleComponent(p, axis) != v[axis])
        return 10;
    error = std::max(error, std::abs(v[0] - (.3 + p[1])));
    error = std::max(error, std::abs(v[1] - .2));
    error = std::max(error, std::abs(v[2] - .1));
  }
  std::cout << "root width=" << width << " affine sampling max error=" << error << '\n';
  if (error > 1e-10)
    return 7;
  if (!field.Advect(.01 * width, moved))
    return 8;
  for (size_t i = 0; i < u.size(); ++i) {
    auto f = mesh.Faces()[i];
    bool interior = true;
    for (double x : f.centre)
      interior &= std::abs(x) < .6 * width;
    if (!interior)
      continue;
    double exact = f.axis == 0 ? .3 + f.centre[1] - .002 * width : f.axis == 1 ? .2 : .1;
    if (std::abs(moved[i] - exact) > 1e-10) {
      std::cerr << "transport error=" << moved[i] - exact << '\n';
      return 9;
    }
  }
  return 0;
}

int main() {
  using namespace wind3d::amr;
  // Non-binary widths exercise rounding at leaf and MAC sample boundaries.
  for (double width : {1., 1.3, 2. / 3.})
    if (int status = Run(width))
      return status;
  // Smooth transport in a uniform region: cubic interpolation should reduce
  // damping without creating extrema. The trace velocity is still linear.
  Topology regular;
  regular.Regrid({2, 2, 2}, 0, 8, [](Block) { return false; });
  Mesh uniform_mesh;
  uniform_mesh.Build(regular, 8, 1, {-1, -1, -1});
  FaceField uniform_field(uniform_mesh, regular, 1, {-1, -1, -1});
  std::vector<double> wave(uniform_mesh.Faces().size()), linear, cubic;
  for (size_t i = 0; i < wave.size(); ++i) {
    auto f = uniform_mesh.Faces()[i];
    wave[i] = f.axis == 0 ? std::sin(2 * f.centre[1]) : f.axis == 1 ? .2 : 0;
  }
  if (!uniform_field.Update(wave) || !uniform_field.Advect(.1, linear) || !uniform_field.Advect(.1, cubic, true))
    return 10;
  double e_linear = 0, e_cubic = 0;
  for (size_t i = 0; i < wave.size(); ++i) {
    auto f = uniform_mesh.Faces()[i];
    if (cubic[i] < -1 - 1e-12 || cubic[i] > 1 + 1e-12)
      return 11;
    if (f.axis != 0 || std::abs(f.centre[1]) > .6)
      continue;
    double exact = std::sin(2 * (f.centre[1] - .02));
    e_linear += std::pow(linear[i] - exact, 2);
    e_cubic += std::pow(cubic[i] - exact, 2);
  }
  std::cout << "smooth transport squared errors linear=" << e_linear << " cubic=" << e_cubic << '\n';
  if (e_cubic >= .2 * e_linear)
    return 12;
  std::cout << "PASS constant preservation and analytic shear translation across refinement levels\n";
}
