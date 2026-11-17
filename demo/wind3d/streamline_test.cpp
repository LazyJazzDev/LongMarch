#include <iostream>

#include "visualization.h"

int main() {
  wind3d::Snapshot s;
  s.nx = 16;
  s.ny = 8;
  s.nz = 12;
  s.controls.speed = 12;
  s.cells.resize(s.nx * s.ny * s.nz);
  for (auto &c : s.cells)
    c.u = 12;
  auto lines = wind3d::visualization::DomainFlowLines(s, 1);
  bool inlet = false, outlet = false, top = false, bottom = false, left = false, right = false;
  for (const auto &v : lines) {
    auto p = v.position;
    if (!std::isfinite(p.x) || p.x < -6.001f || p.x > 10.001f || std::abs(p.y) > 4.001f || std::abs(p.z) > 6.001f)
      return 1;
    inlet |= p.x < -5.99f;
    outlet |= p.x > 9.99f;
    top |= p.y > 3.5f;
    bottom |= p.y < -3.5f;
    left |= p.z < -5.5f;
    right |= p.z > 5.5f;
  }
  if (!(inlet && outlet && top && bottom && left && right))
    return 2;
  for (size_t i = 0; i + 1 < lines.size(); i += 2)
    if (lines[i].position.y != lines[i + 1].position.y || lines[i].position.z != lines[i + 1].position.z)
      return 3;
  for (auto &c : s.cells)
    c = {};
  if (!wind3d::visualization::DomainFlowLines(s, 1).empty())
    return 4;
  std::cout << "PASS: full-domain uniform flow, straight segments, bounds and stagnation\n";
}
