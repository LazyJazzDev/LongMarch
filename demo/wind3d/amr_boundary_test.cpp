#include <algorithm>
#include <cmath>
#include <iostream>

#include "aircraft.h"
#include "amr_projection.h"

bool ValidateThinConnectivity() {
  using namespace wind3d::amr;
  Topology tree;
  if (!tree.Regrid({1, 1, 1}, 0, 1, [](Block) { return false; }))
    return false;
  int leaks = 0, total = 0;
  for (double width : {1., 1.3})
    for (int plane = 0; plane < 3; ++plane)
      for (int angle = 0; angle <= 90; angle += 15)
        for (int phase = 0; phase < 32; ++phase) {
          double n[3]{};
          n[plane] = std::cos(angle * std::acos(-1.) / 180);
          n[(plane + 1) % 3] = std::sin(angle * std::acos(-1.) / 180);
          double offset = width * (.5 * (n[0] + n[1] + n[2]) + (phase / 32. - .5) * .125);
          auto side = [&](std::array<double, 3> p) { return n[0] * p[0] + n[1] * p[1] + n[2] * p[2] - offset; };
          Mesh mesh;
          if (!mesh.Build(tree, 8, width) ||
              !mesh.Embed([&](std::array<double, 3> p) { return std::abs(side(p)) - .0125 * width; }))
            return false;
          int start = 0, end = 0;
          for (int i = 0; i < int(mesh.Cells().size()); ++i) {
            if (side(mesh.Cells()[i].centre) < side(mesh.Cells()[start].centre))
              start = i;
            if (side(mesh.Cells()[i].centre) > side(mesh.Cells()[end].centre))
              end = i;
          }
          std::vector<int> q{start};
          std::vector<bool> seen(mesh.Cells().size());
          seen[start] = true;
          for (size_t k = 0; k < q.size(); ++k) {
            int c = q[k];
            for (int j = mesh.Offsets()[c]; j < mesh.Offsets()[c + 1]; ++j) {
              auto f = mesh.Faces()[std::abs(mesh.Incidence()[j]) - 1];
              int next = f.low == c ? f.high : f.low;
              if (f.aperture > 0 && next >= 0 && !seen[next]) {
                seen[next] = true;
                q.push_back(next);
              }
            }
          }
          ++total;
          if (seen[end]) {
            ++leaks;
            if (leaks <= 8)
              std::cout << "leak plane=" << plane << " angle=" << angle << " phase=" << phase << '\n';
          }
        }
  std::cout << "leaks=" << leaks << "/" << total << '\n';
  return leaks == 0;
}

bool ValidateReuse() {
  using namespace wind3d::amr;
  for (double width : {1.0, 1.3}) {
    Topology fine, mixed;
    if (!fine.Regrid({2, 2, 2}, 1, 64, [](Block) { return true; }) ||
        !mixed.Regrid({2, 2, 2}, 1, 64, [](Block b) { return b.x == 0; }))
      return false;
    Mesh previous;
    size_t calls = 0;
    auto distance = [&](std::array<double, 3> p) {
      ++calls;
      return std::max({std::abs(p[0] + .5 * width), std::abs(p[1]), std::abs(p[2])}) - .21 * width;
    };
    if (!previous.Build(fine, 4, width, {-width, -width, -width}) || !previous.Embed(distance))
      return false;
    // Coarsen and refine away from the fixed fine body region. Verify every
    // aperture and active cell, including unchanged centres with changed neighbors.
    for (const auto *tree : {&mixed, &fine}) {
      Mesh fresh, reused;
      if (!fresh.Build(*tree, 4, width, {-width, -width, -width}) ||
          !reused.Build(*tree, 4, width, {-width, -width, -width}))
        return false;
      calls = 0;
      if (!fresh.Embed(distance))
        return false;
      size_t full_calls = calls;
      calls = 0;
      if (!reused.Embed(distance, &previous) || calls >= full_calls)
        return false;
      for (size_t i = 0; i < fresh.Faces().size(); ++i)
        if (fresh.Faces()[i].aperture != reused.Faces()[i].aperture)
          return false;
      for (size_t i = 0; i < fresh.Cells().size(); ++i)
        if (fresh.Cells()[i].active != reused.Cells()[i].active)
          return false;
      std::cout << "embedding width=" << width << " SDF calls=" << calls << '/' << full_calls << '\n';
      previous = std::move(reused);
    }
    // Rebuilding invalidates the cached embedding; an unembedded mesh's default
    // open faces must never be accepted as sampled geometry.
    if (!previous.Build(fine, 4, width, {-width, -width, -width}))
      return false;
    Mesh check;
    if (!check.Build(fine, 4, width, {-width, -width, -width}) || !check.Embed(distance, &previous))
      return false;
    bool closed = false;
    for (const auto &face : check.Faces())
      closed |= face.aperture == 0;
    if (!closed)
      return false;
  }
  return true;
}

int main() {
  using namespace wind3d::amr;
  if (!ValidateThinConnectivity())
    return 18;
  if (!ValidateReuse())
    return 17;
  Topology tree;
  if (!tree.Regrid({4, 4, 4}, 1, 1000,
                   [](Block b) { return b.x >= 1 && b.x <= 2 && b.y >= 1 && b.y <= 2 && b.z >= 1 && b.z <= 2; }))
    return 1;
  Mesh mesh;
  if (!mesh.Build(tree, 4, 1, {-2, -2, -2}))
    return 2;
  auto box = [](std::array<double, 3> p) { return std::max({std::abs(p[0]), std::abs(p[1]), std::abs(p[2])}) - .61; };
  if (!mesh.Embed(box))
    return 3;
  size_t partial = 0, closed = 0, inactive = 0;
  for (auto &f : mesh.Faces()) {
    partial += f.aperture > 0 && f.aperture < 1;
    closed += f.aperture == 0;
  }
  for (auto &c : mesh.Cells())
    inactive += !c.active;
  if (!partial || !closed || !inactive)
    return 4;
  std::vector<double> pressure(mesh.Cells().size(), 100);
  std::array<double, 6> loads{};
  if (!mesh.PressureLoads(pressure, loads))
    return 5;
  for (double f : loads)
    if (std::abs(f) > 1e-10)
      return 6;
  FaceGradient g;
  if (!g.Build(mesh))
    return 7;
  std::vector<double> u(mesh.Faces().size()), p;
  for (size_t i = 0; i < u.size(); ++i)
    u[i] = mesh.Faces()[i].axis == 0 ? 1 : 0;
  auto solve = ProjectReference(mesh, g, u, p, 1e-8, 2000);
  std::cout << "embedded box partial=" << partial << " closed=" << closed << " inactive=" << inactive
            << " iterations=" << solve.iterations << " residual=" << solve.relative_residual << '\n';
  if (!solve.converged)
    return 8;
  for (size_t i = 0; i < u.size(); ++i)
    if (mesh.Faces()[i].aperture == 0 && u[i] != 0)
      return 9;
  // A boundary crossing a refinement interface must be rejected, not silently
  // change the body representation to a coarse approximation.
  auto before = mesh.Faces();
  if (mesh.Embed([](std::array<double, 3> p) { return p[0] + .95; }))
    return 10;
  for (size_t i = 0; i < before.size(); ++i)
    if (before[i].aperture != mesh.Faces()[i].aperture)
      return 11;
  // Actual aircraft geometry, with all components inside a uniform fine region.
  Topology aircraft_tree;
  if (!aircraft_tree.Regrid({8, 4, 6}, 0, 192, [](Block) { return false; }))
    return 12;
  Mesh aircraft;
  if (!aircraft.Build(aircraft_tree, 4, 2, {-6, -4, -6}))
    return 13;
  wind3d::Controls controls;
  controls.angle = 4;
  controls.flap = 20;
  controls.elevator = -10;
  if (!aircraft.Embed([&](std::array<double, 3> p) {
        return double(wind3d::AircraftDistance({float(p[0]), float(p[1]), float(p[2])}, controls));
      }))
    return 14;
  pressure.assign(aircraft.Cells().size(), 100);
  if (!aircraft.PressureLoads(pressure, loads))
    return 15;
  for (double f : loads)
    if (std::abs(f) > 1e-8)
      return 16;
  std::cout
      << "PASS stationary cut-face projection, closed-surface aircraft force/moment cancellation and interface rollback\n";
}
