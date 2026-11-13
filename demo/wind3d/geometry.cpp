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
      for (int part = 0; part < 7; ++part) {
        if (part == 5 && side == -1)
          continue;
        Vec3 color = part == 1   ? Vec3{0.72f, 0.66f, 0.52f}
                     : part == 2 ? Vec3{0.45f, 0.61f, 0.72f}
                     : part == 4 ? Vec3{0.48f, 0.65f, 0.59f}
                                 : Vec3{0.35f, 0.49f, 0.69f};
        Foil(AircraftPart(part, side, controls_), color, part == 5 ? Vec3{0.035f, 0.15f, 0.32f} : skin);
      }
      Nacelle(side);
      CabinDetails(side);
    }
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

  Vertex BodyVertex(float x, float phi, Vec3 color, float offset = 0) const {
    Vec3 p = FuselagePoint(x, phi);
    Vec3 dx =
        Add(FuselagePoint(std::min(3.0f, x + 0.001f), phi), Scale(FuselagePoint(std::max(-3.0f, x - 0.001f), phi), -1));
    Vec3 dp = Add(FuselagePoint(x, phi + 0.001f), Scale(FuselagePoint(x, phi - 0.001f), -1));
    Vec3 n = Unit(Cross(dp, dx));
    if (std::abs(x) >= 3)
      n = {x < 0 ? -1.0f : 1.0f, 0, 0};
    return MakeVertex(Add(p, Scale(n, offset)), n, color);
  }

  void Fuselage() {
    constexpr int rings = 160, slices = 96;
    auto point = [&](int ring, int segment) {
      float x = -3 + 6.0f * ring / rings, phi = 2 * pi * segment / slices;
      Vec3 color = cosf(phi) < -0.65f ? Vec3{0.045f, 0.13f, 0.24f} : Vec3{0.86f, 0.88f, 0.90f};
      return BodyVertex(x, phi, color);
    };
    for (int i = 0; i < rings; ++i)
      for (int j = 0; j < slices; ++j)
        Quad(point(i, j), point(i + 1, j), point(i + 1, (j + 1) % slices), point(i, (j + 1) % slices));
  }

  void CabinDetails(int side) {
    // Paint/flush glazing lies on the explicit fuselage; it adds no fake solid
    // obstacles to the aerodynamic boundary. Small offsets avoid depth fighting.
    const Vec3 glass{0.015f, 0.045f, 0.075f}, trim{0.20f, 0.26f, 0.31f};
    auto patch = [&](float x, float phi, float rx, float rp) {
      auto center = BodyVertex(x, side * phi, glass, 0.001f);
      for (int i = 0; i < 24; ++i) {
        float a = 2 * pi * i / 24, b = 2 * pi * (i + 1) / 24;
        Triangle(center, BodyVertex(x + rx * cosf(a), side * (phi + rp * sinf(a)), glass, 0.001f),
                 BodyVertex(x + rx * cosf(b), side * (phi + rp * sinf(b)), glass, 0.001f));
      }
    };
    for (int i = 0; i < 23; ++i)
      patch(-1.75f + i * 0.14f, 1.17f, 0.029f, 0.11f);
    // Tessellated trapezoidal cockpit panes follow the nose skin. A single
    // planar quad would sink into the convex fuselage and disappear.
    auto pane = [&](Vec3 a, Vec3 b, Vec3 c, Vec3 d) {
      auto point = [&](int i, int j) {
        float u = i / 8.0f, v = j / 8.0f;
        Vec3 q = Add(Scale(Add(Scale(a, 1 - u), Scale(b, u)), 1 - v), Scale(Add(Scale(d, 1 - u), Scale(c, u)), v));
        return BodyVertex(q.x, side * q.y, glass, 0.001f);
      };
      for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
          Quad(point(i, j), point(i + 1, j), point(i + 1, j + 1), point(i, j + 1));
    };
    pane({-2.70f, 0.28f, 0}, {-2.52f, 0.34f, 0}, {-2.50f, 0.72f, 0}, {-2.69f, 0.68f, 0});
    pane({-2.65f, 0.76f, 0}, {-2.48f, 0.80f, 0}, {-2.39f, 1.12f, 0}, {-2.58f, 1.10f, 0});
    pane({-2.54f, 1.14f, 0}, {-2.37f, 1.16f, 0}, {-2.24f, 1.37f, 0}, {-2.43f, 1.40f, 0});
    for (float x : {-1.96f, 1.63f}) {
      for (int i = 0; i < 32; ++i) {
        float a = 2 * pi * i / 32, b = 2 * pi * (i + 1) / 32;
        auto outline = [&](float t, float scale) {
          return BodyVertex(x + 0.055f * scale * std::copysign(std::pow(std::abs(cosf(t)), 0.35f), cosf(t)),
                            side * (1.6f + 0.58f * scale * std::copysign(std::pow(std::abs(sinf(t)), 0.35f), sinf(t))),
                            trim, 0.0012f);
        };
        Quad(outline(a, 1), outline(b, 1), outline(b, 0.94f), outline(a, 0.94f));
      }
    }
  }

  void Nacelle(int side) {
    constexpr int rings = 32, slices = 64;
    auto point = [&](float t, float phi) {
      Vec3 p = NacellePoint(t, phi, side);
      float derivative = (NacelleRadius(t + 0.0001f) - NacelleRadius(t - 0.0001f)) / 0.00018f;
      Vec3 color = t < 0.08f ? Vec3{0.48f, 0.51f, 0.55f} : skin;
      return MakeVertex(p, {-derivative, cosf(phi), sinf(phi)}, color);
    };
    for (int i = 0; i < rings; ++i)
      for (int j = 0; j < slices; ++j)
        Quad(point(float(i) / rings, 2 * pi * j / slices), point(float(i + 1) / rings, 2 * pi * j / slices),
             point(float(i + 1) / rings, 2 * pi * (j + 1) / slices),
             point(float(i) / rings, 2 * pi * (j + 1) / slices));
    for (int end : {0, 1}) {
      Vec3 center{-1.3f + 0.9f * end, -0.54f, side * 0.92f}, normal{end ? 1.0f : -1.0f, 0, 0};
      for (int j = 0; j < slices; ++j) {
        Vec3 color = !end && j % 4 == 0 ? Vec3{0.10f, 0.12f, 0.14f} : Vec3{0.016f, 0.020f, 0.025f};
        Triangle(MakeVertex(center, normal, color),
                 MakeVertex(NacellePoint(float(end), 2 * pi * j / slices, side), normal, color),
                 MakeVertex(NacellePoint(float(end), 2 * pi * (j + 1) / slices, side), normal, color));
      }
    }
  }

  void Foil(const FoilPart &part, Vec3 control_color, Vec3 fixed_color) {
    float start = part.begin, end = part.end;
    // Separate closed fixed and hinged solids retain their thin trailing edges
    // at every deflection. Spanwise caps close tips and flap/aileron boundaries.
    constexpr int chord_segments = 48, span_segments = 16;
    for (int piece = 0; piece < (part.hinged ? 2 : 1); ++piece) {
      float first = piece ? 0.72f : 0, last = piece || !part.hinged ? 1 : 0.72f;
      Vec3 color = piece ? control_color : fixed_color;
      auto position = [&](float u, float s, int surface) { return FoilPoint(part, u, s, surface, piece != 0); };
      auto fraction = [&](int i) { return first + (last - first) * (1 - cosf(pi * i / chord_segments)) * 0.5f; };
      auto vertex = [&](float u, float s, int surface) {
        constexpr float e = 0.0001f;
        Vec3 du =
            Add(position(std::min(last, u + e), s, surface), Scale(position(std::max(first, u - e), s, surface), -1));
        Vec3 ds = Add(position(u, s + e, surface), Scale(position(u, s - e, surface), -1));
        Vec3 n = Scale(Cross(ds, du), surface * Dot(Cross(part.spec.x_axis, part.spec.y_axis), part.spec.z_axis));
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
        Vec3 n =
            Add(Cross(Add(b, Scale(a, -1)), Add(c, Scale(a, -1))), Cross(Add(c, Scale(a, -1)), Add(d, Scale(a, -1))));
        if (piece) {
          Vec3 origin = ControlTransform(part, {}, false);
          outward = Add(ControlTransform(part, outward, false), Scale(origin, -1));
        }
        outward = FoilVector(part.spec, outward);
        if (n.x * outward.x + n.y * outward.y + n.z * outward.z < 0)
          n = Scale(n, -1);
        Quad(MakeVertex(a, n, color), MakeVertex(b, n, color), MakeVertex(c, n, color), MakeVertex(d, n, color));
      };
      for (float s : {start, end})
        for (int i = 0; i < chord_segments; ++i)
          cap(position(fraction(i), s, -1), position(fraction(i + 1), s, -1), position(fraction(i + 1), s, 1),
              position(fraction(i), s, 1), {0, 0, s == start ? -1.0f : 1.0f});
      // Exposed hinge faces, rather than an open gap when the control rotates.
      for (float u : {first, last})
        for (int j = 0; j < span_segments; ++j) {
          float a = start + (end - start) * j / span_segments, b = start + (end - start) * (j + 1) / span_segments;
          Vec3 outward{u == first ? -1.0f : 1.0f, 0, 0};
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
