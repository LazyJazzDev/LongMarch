#include <array>

#include "simulation.h"

namespace wind3d {
std::vector<Vertex> AircraftMesh(const Controls &controls) {
  constexpr int nx = 96, ny = 72, nz = 120;
  constexpr float h = 0.08f;
  auto index = [=](int x, int y, int z) { return (z * (ny + 1) + y) * (nx + 1) + x; };
  auto position = [=](int x, int y, int z) { return Vec3{(x - nx / 2) * h, (y - ny / 2) * h, (z - nz / 2) * h}; };
  std::vector<float> distance((nx + 1) * (ny + 1) * (nz + 1));
  for (int z = 0; z <= nz; ++z)
    for (int y = 0; y <= ny; ++y)
      for (int x = 0; x <= nx; ++x)
        distance[index(x, y, z)] = AircraftDistance(position(x, y, z), controls);
  std::vector<Vertex> vertices;
  auto vertex = [&](Vec3 p) {
    constexpr float e = 0.001f;
    Vec3 normal{AircraftDistance(Add(p, {e, 0, 0}), controls) - AircraftDistance(Add(p, {-e, 0, 0}), controls),
                AircraftDistance(Add(p, {0, e, 0}), controls) - AircraftDistance(Add(p, {0, -e, 0}), controls),
                AircraftDistance(Add(p, {0, 0, e}), controls) - AircraftDistance(Add(p, {0, 0, -e}), controls)};
    normal = Scale(normal, 1 / fmaxf(Length(normal), 1e-8f));
    Vec3 body = BodyPoint(p, controls);
    Vec3 color{0.72f, 0.79f, 0.87f};
    float s = fabsf(body.z);
    float chord = 1.8f + (0.55f - 1.8f) * fminf(s / 4.5f, 1.0f);
    if (s > 0.4f && body.x > -1.05f + 0.38f * s + 0.7f * chord && body.x < 1.7f)
      color = s < 2.7f ? Vec3{0.95f, 0.53f, 0.13f} : Vec3{0.17f, 0.65f, 0.95f};
    if (body.x > 2.3f && s > 0.3f)
      color = {0.26f, 0.78f, 0.54f};
    if (body.y > 0.5f && body.x > 2.1f)
      color = {0.77f, 0.38f, 0.85f};
    vertices.push_back({p, normal, color});
  };
  constexpr int tetrahedra[6][4] = {{0, 1, 3, 7}, {0, 3, 2, 7}, {0, 2, 6, 7}, {0, 6, 4, 7}, {0, 4, 5, 7}, {0, 5, 1, 7}};
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < nx; ++x) {
        Vec3 p[8];
        float d[8];
        int negative = 0;
        for (int i = 0; i < 8; ++i) {
          p[i] = position(x + (i & 1), y + ((i >> 1) & 1), z + ((i >> 2) & 1));
          d[i] = distance[index(x + (i & 1), y + ((i >> 1) & 1), z + ((i >> 2) & 1))];
          negative += d[i] < 0;
        }
        if (!negative || negative == 8)
          continue;
        auto cut = [&](int a, int b) {
          float t = d[a] / (d[a] - d[b]);
          return Add(Scale(p[a], 1 - t), Scale(p[b], t));
        };
        for (const auto &tet : tetrahedra) {
          int in[4], out[4], ni = 0, no = 0;
          for (int i : tet) {
            if (d[i] < 0)
              in[ni++] = i;
            else
              out[no++] = i;
          }
          if (ni == 1)
            for (int j = 0; j < 3; ++j)
              vertex(cut(in[0], out[j]));
          if (ni == 3)
            for (int j = 0; j < 3; ++j)
              vertex(cut(out[0], in[j]));
          if (ni == 2) {
            Vec3 a = cut(in[0], out[0]), b = cut(in[0], out[1]), c = cut(in[1], out[1]), d2 = cut(in[1], out[0]);
            vertex(a);
            vertex(b);
            vertex(c);
            vertex(a);
            vertex(c);
            vertex(d2);
          }
        }
      }
  return vertices;
}
}  // namespace wind3d
