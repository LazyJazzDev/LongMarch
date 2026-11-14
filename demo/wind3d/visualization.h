#pragma once
#include <algorithm>

#include "simulation.h"

namespace wind3d::visualization {
inline Vec3 SampleVelocity(const Snapshot &snapshot, Vec3 p) {
  float h = 16.0f / snapshot.nx;
  float gx = (p.x + 6) / h - 0.5f, gy = (p.y + 4) / h - 0.5f, gz = (p.z + 6) / h - 0.5f;
  int x = int(floorf(gx)), y = int(floorf(gy)), z = int(floorf(gz));
  Vec3 velocity{};
  for (int k = 0; k < 2; ++k)
    for (int j = 0; j < 2; ++j)
      for (int i = 0; i < 2; ++i) {
        int a = std::clamp(x + i, 0, snapshot.nx - 1), b = std::clamp(y + j, 0, snapshot.ny - 1),
            c = std::clamp(z + k, 0, snapshot.nz - 1);
        const auto &cell = snapshot.cells[(c * snapshot.ny + b) * snapshot.nx + a];
        float weight = (i ? gx - x : 1 - gx + x) * (j ? gy - y : 1 - gy + y) * (k ? gz - z : 1 - gz + z);
        velocity = Add(velocity, Scale({cell.u, cell.v, cell.w}, weight));
      }
  return velocity;
}

inline void Line(std::vector<Vertex> &lines, Vec3 a, Vec3 b, Vec3 color) {
  lines.push_back({a, {}, color});
  lines.push_back({b, {}, color});
}

inline void Arrow(std::vector<Vertex> &lines, Vec3 origin, Vec3 vector, Vec3 color) {
  float length = Length(vector);
  if (length < 1e-6f)
    return;
  Vec3 direction = Scale(vector, 1 / length);
  Vec3 perpendicular = Cross(direction, fabsf(direction.y) < 0.9f ? Vec3{0, 1, 0} : Vec3{0, 0, 1});
  perpendicular = Scale(perpendicular, 0.12f / Length(perpendicular));
  Vec3 end = Add(origin, vector), base = Add(end, Scale(direction, -0.28f));
  Line(lines, origin, end, color);
  Line(lines, end, Add(base, perpendicular), color);
  Line(lines, end, Add(base, Scale(perpendicular, -1)), color);
}

inline std::vector<Vertex> FlowLines(const Snapshot &snapshot, bool streamlines, bool loads, int density = 1) {
  std::vector<Vertex> lines;
  for (int x = -5; x <= 9; ++x)
    Line(lines, {float(x), -2, -5}, {float(x), -2, 5}, {0.055f, 0.08f, 0.12f});
  for (int z = -5; z <= 5; ++z)
    Line(lines, {-5, -2, float(z)}, {9, -2, float(z)}, {0.055f, 0.08f, 0.12f});
  Arrow(lines, {-4, -1.8f, -4.7f}, {1, 0, 0}, {1, 0.25f, 0.2f});
  Arrow(lines, {-4, -1.8f, -4.7f}, {0, 1, 0}, {0.2f, 1, 0.3f});
  Arrow(lines, {-4, -1.8f, -4.7f}, {0, 0, 1}, {0.2f, 0.5f, 1});
  if (streamlines) {
    density = std::clamp(density, 1, 4);
    for (int iz = -10 * density; iz <= 10 * density; ++iz)
      for (int iy = -density; iy <= 2 * density; ++iy) {
        Vec3 p{-4.5f, iy * (0.32f / density) + 0.05f, iz * (0.30f / density)};
        for (int step = 0; step < 170; ++step) {
          Vec3 velocity = SampleVelocity(snapshot, p);
          float speed = Length(velocity);
          if (speed < 0.05f)
            break;
          Vec3 mid = Add(p, Scale(velocity, 0.04f / speed));
          Vec3 v2 = SampleVelocity(snapshot, mid);
          Vec3 next = Add(p, Scale(v2, 0.08f / std::max(Length(v2), 0.05f)));
          if (next.x > 9 || fabsf(next.y) > 3.7f || fabsf(next.z) > 5.7f ||
              AircraftDistance(next, snapshot.controls) < 0)
            break;
          float ratio = std::clamp(speed / snapshot.controls.speed * 0.6f, 0.0f, 1.0f);
          Vec3 color{0.05f + ratio * ratio * 0.8f, 0.25f + 0.6f * ratio, 0.9f - 0.75f * ratio};
          Line(lines, p, next, color);
          p = next;
        }
      }
  }
  if (loads) {
    Vec3 origin{0, 2, 0};
    Line(lines, {}, origin, {0.3f, 0.35f, 0.4f});
    Arrow(lines, origin, Scale(snapshot.force, 3 / (100 + Length(snapshot.force))), {0.1f, 1, 0.8f});
    Arrow(lines, origin, Scale(snapshot.moment, 3 / (100 + Length(snapshot.moment))), {1, 0.25f, 0.85f});
  }
  return lines;
}
}  // namespace wind3d::visualization
