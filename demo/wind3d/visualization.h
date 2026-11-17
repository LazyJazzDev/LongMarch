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

// Volumetric seeds catch disconnected recirculation as well as inlet flow.
// Occupied seed bins prevent tracing the same region repeatedly.
inline std::vector<Vertex> DomainFlowLines(const Snapshot &snapshot, int density) {
  std::vector<Vertex> lines;
  if (snapshot.cells.empty())
    return lines;
  float spacing = .75f / std::clamp(density, 1, 4);
  int nx = int(std::ceil(16 / spacing)), ny = int(std::ceil(8 / spacing)), nz = int(std::ceil(12 / spacing));
  std::vector<int> occupied(size_t(nx) * ny * nz, -1);
  auto inside = [](Vec3 p) { return p.x >= -6 && p.x <= 10 && p.y >= -4 && p.y <= 4 && p.z >= -6 && p.z <= 6; };
  auto bin = [&](Vec3 p) {
    int x = std::clamp(int((p.x + 6) * nx / 16), 0, nx - 1);
    int y = std::clamp(int((p.y + 4) * ny / 8), 0, ny - 1);
    int z = std::clamp(int((p.z + 6) * nz / 12), 0, nz - 1);
    return (z * ny + y) * nx + x;
  };
  int owner = 0;
  for (int ix = 0; ix < nx; ++ix)
    for (int iz = 0; iz < nz; ++iz)
      for (int iy = 0; iy < ny; ++iy) {
        Vec3 seed{-6 + (ix + .5f) * 16 / nx, -4 + (iy + .5f) * 8 / ny, -6 + (iz + .5f) * 12 / nz};
        if (occupied[bin(seed)] >= 0 || AircraftDistance(seed, snapshot.controls) < 0)
          continue;
        occupied[bin(seed)] = owner;
        auto trace = [&](float sign) {
          std::vector<Vec3> points;
          Vec3 p = seed;
          for (int step = 0; step < 600; ++step) {
            Vec3 v = SampleVelocity(snapshot, p);
            float speed = Length(v);
            if (!std::isfinite(speed) || speed < .05f)
              break;
            Vec3 mid = Add(p, Scale(v, sign * .04f / speed));
            Vec3 vm = SampleVelocity(snapshot, mid);
            float sm = Length(vm);
            if (!std::isfinite(sm) || sm < .05f)
              break;
            Vec3 next = Add(p, Scale(vm, sign * .08f / sm));
            bool boundary = !inside(next);
            if (boundary) {
              Vec3 d = Add(next, Scale(p, -1));
              float t = 1;
              auto clip = [&](float a, float b, float lo, float hi) {
                if (b > 0)
                  t = std::min(t, (hi - a) / b);
                if (b < 0)
                  t = std::min(t, (lo - a) / b);
              };
              clip(p.x, d.x, -6, 10);
              clip(p.y, d.y, -4, 4);
              clip(p.z, d.z, -6, 6);
              next = Add(p, Scale(d, std::max(0.f, t)));
            }
            if (AircraftDistance(mid, snapshot.controls) < 0 || AircraftDistance(next, snapshot.controls) < 0)
              break;
            int cell = bin(next);
            if (occupied[cell] >= 0 && occupied[cell] != owner)
              break;
            occupied[cell] = owner;
            if (Length(Add(next, Scale(p, -1))) < 1e-6f)
              break;
            points.push_back(next);
            p = next;
            if (boundary)
              break;
          }
          return points;
        };
        auto backward = trace(-1), forward = trace(1);
        std::reverse(backward.begin(), backward.end());
        backward.push_back(seed);
        backward.insert(backward.end(), forward.begin(), forward.end());
        for (size_t i = 1; i < backward.size(); ++i) {
          float ratio =
              std::clamp(Length(SampleVelocity(snapshot, backward[i - 1])) / snapshot.controls.speed * .6f, 0.f, 1.f);
          Line(lines, backward[i - 1], backward[i],
               {.05f + ratio * ratio * .8f, .25f + .6f * ratio, .9f - .75f * ratio});
        }
        ++owner;
      }
  return lines;
}

inline std::vector<Vertex> FlowLines(const Snapshot &snapshot,
                                     bool streamlines,
                                     bool loads,
                                     int density = 1,
                                     bool decorations = true) {
  std::vector<Vertex> lines;
  if (decorations) {
    for (int x = -5; x <= 9; ++x)
      Line(lines, {float(x), -2, -5}, {float(x), -2, 5}, {0.055f, 0.08f, 0.12f});
    for (int z = -5; z <= 5; ++z)
      Line(lines, {-5, -2, float(z)}, {9, -2, float(z)}, {0.055f, 0.08f, 0.12f});
    Arrow(lines, {-4, -1.8f, -4.7f}, {1, 0, 0}, {1, 0.25f, 0.2f});
    Arrow(lines, {-4, -1.8f, -4.7f}, {0, 1, 0}, {0.2f, 1, 0.3f});
    Arrow(lines, {-4, -1.8f, -4.7f}, {0, 0, 1}, {0.2f, 0.5f, 1});
  }
  if (streamlines && !snapshot.cells.empty()) {
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
