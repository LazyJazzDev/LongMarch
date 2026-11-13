#include <algorithm>

#include "simulation.h"

namespace wind3d {
namespace {
constexpr float pi = 3.14159265359f;
const Vec3 skin{0.72f, 0.79f, 0.87f};

Vec3 Normalize(Vec3 v) {
  return Scale(v, 1 / std::max(Length(v), 1e-12f));
}

// Explicit parametric patches. No volume sampling, implicit zero crossing or
// marching tetrahedra is involved in rendering the aircraft.
class MeshBuilder {
 public:
  explicit MeshBuilder(const Controls &controls) : controls_(controls) {
  }

  std::vector<Vertex> Build() {
    Fuselage();
    for (int side : {-1, 1}) {
      Foil(0, 2.7f, side, 4.5f, 1.8f, 0.55f, -1.05f, 0.38f, 0.045f, 0, controls_.flap, false, {0.95f, 0.53f, 0.13f});
      Foil(2.7f, 4.5f, side, 4.5f, 1.8f, 0.55f, -1.05f, 0.38f, 0.045f, 0, side * controls_.aileron, false,
           {0.17f, 0.65f, 0.95f});
      Foil(0, 1.65f, side, 1.65f, 0.95f, 0.4f, 1.65f, 0.32f, 0.05f, 0.18f, controls_.elevator, false,
           {0.26f, 0.78f, 0.54f});
    }
    Foil(0, 1.55f, 1, 1.55f, 1.25f, 0.45f, 1.35f, 0.45f, 0, 0.1f, controls_.rudder, true, {0.77f, 0.38f, 0.85f});
    return std::move(vertices_);
  }

 private:
  Vertex MakeVertex(Vec3 p, Vec3 n, Vec3 color) const {
    return {RotateZ(p, -controls_.angle), RotateZ(Normalize(n), -controls_.angle), color};
  }

  void Triangle(Vertex a, Vertex b, Vertex c) {
    Vec3 normal = Cross(Add(b.position, Scale(a.position, -1)), Add(c.position, Scale(a.position, -1)));
    if (Length(normal) < 1e-10f)
      return;  // Collapsed pole/trailing-edge triangle.
    Vec3 outward = Add(Add(a.normal, b.normal), c.normal);
    if (normal.x * outward.x + normal.y * outward.y + normal.z * outward.z < 0)
      std::swap(b, c);
    vertices_.insert(vertices_.end(), {a, b, c});
  }

  void Quad(Vertex a, Vertex b, Vertex c, Vertex d) {
    Triangle(a, b, c);
    Triangle(a, c, d);
  }

  void Fuselage() {
    auto point = [&](int ring, int segment) {
      float t = pi * ring / 96, phi = 2 * pi * segment / 64;
      Vec3 p{-3 * cosf(t), 0.38f * sinf(t) * cosf(phi), 0.38f * sinf(t) * sinf(phi)};
      if (ring == 0 || ring == 96)
        p = {ring == 0 ? -3.0f : 3.0f, 0, 0};
      return MakeVertex(p, {p.x / 9, p.y / (0.38f * 0.38f), p.z / (0.38f * 0.38f)}, skin);
    };
    for (int ring = 0; ring < 96; ++ring)
      for (int segment = 0; segment < 64; ++segment)
        Quad(point(ring, segment), point(ring + 1, segment), point(ring + 1, (segment + 1) % 64),
             point(ring, (segment + 1) % 64));
  }

  void Foil(float start,
            float end,
            int side,
            float span,
            float root,
            float tip,
            float leading,
            float sweep,
            float dihedral,
            float offset,
            float deflection,
            bool vertical,
            Vec3 control_color) {
    // Separate closed fixed and hinged solids retain their thin trailing edges
    // at every deflection. Spanwise caps close tips and flap/aileron boundaries.
    constexpr int chord_segments = 48, span_segments = 16;
    for (int piece = 0; piece < 2; ++piece) {
      float first = piece ? 0.72f : 0, last = piece ? 1 : 0.72f;
      Vec3 color = piece ? control_color : skin;
      auto transform = [&](Vec3 p) { return vertical ? Vec3{p.x, p.z + offset, p.y} : Vec3{p.x, p.y + offset, p.z}; };
      auto position = [&](float u, float s, int surface) {
        float chord = root + (tip - root) * s / span;
        Vec3 local{(u - 0.72f) * chord, surface * FoilThickness(u, chord), 0};
        if (piece)
          local = RotateZ(local, -deflection);
        return transform({leading + sweep * s + 0.72f * chord + local.x, dihedral * s + local.y, side * s});
      };
      auto fraction = [&](int i) { return first + (last - first) * (1 - cosf(pi * i / chord_segments)) * 0.5f; };
      auto vertex = [&](float u, float s, int surface) {
        constexpr float e = 0.0001f;
        Vec3 du =
            Add(position(std::min(last, u + e), s, surface), Scale(position(std::max(first, u - e), s, surface), -1));
        Vec3 ds = Add(position(u, s + e, surface), Scale(position(u, s - e, surface), -1));
        Vec3 n = Scale(Cross(ds, du), float(side * surface * (vertical ? -1 : 1)));
        return MakeVertex(position(u, s, surface), n, color);
      };
      for (int surface : {-1, 1}) {
        for (int j = 0; j < span_segments; ++j) {
          float a = start + (end - start) * j / span_segments, b = start + (end - start) * (j + 1) / span_segments;
          for (int i = 0; i < chord_segments; ++i) {
            float u = fraction(i), v = fraction(i + 1);
            Quad(vertex(u, a, surface), vertex(v, a, surface), vertex(v, b, surface), vertex(u, b, surface));
          }
        }
      }
      auto cap = [&](Vec3 a, Vec3 b, Vec3 c, Vec3 d, Vec3 outward) {
        Vec3 n = Cross(Add(b, Scale(a, -1)), Add(d, Scale(a, -1)));
        if (vertical)
          outward = {outward.x, outward.z, outward.y};
        if (n.x * outward.x + n.y * outward.y + n.z * outward.z < 0)
          n = Scale(n, -1);
        Quad(MakeVertex(a, n, color), MakeVertex(b, n, color), MakeVertex(c, n, color), MakeVertex(d, n, color));
      };
      for (float s : {start, end})
        for (int i = 0; i < chord_segments; ++i)
          cap(position(fraction(i), s, -1), position(fraction(i + 1), s, -1), position(fraction(i + 1), s, 1),
              position(fraction(i), s, 1), {0, 0, float(side * (s == start ? -1 : 1))});
      // Exposed hinge faces, rather than an open gap when the control rotates.
      for (float u : {first, last})
        for (int j = 0; j < span_segments; ++j) {
          float a = start + (end - start) * j / span_segments, b = start + (end - start) * (j + 1) / span_segments;
          Vec3 outward = RotateZ({u == first ? -1.0f : 1.0f, 0, 0}, piece ? -deflection : 0);
          cap(position(u, a, -1), position(u, b, -1), position(u, b, 1), position(u, a, 1), outward);
        }
    }
  }

  Controls controls_;
  std::vector<Vertex> vertices_;
};
}  // namespace

std::vector<Vertex> AircraftMesh(const Controls &controls) {
  return MeshBuilder(controls).Build();
}
}  // namespace wind3d
