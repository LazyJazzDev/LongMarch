#include <algorithm>

#include "simulation.h"

namespace wind3d {
namespace {
constexpr float pi = 3.14159265359f;
const Vec3 skin{0.78f, 0.78f, 0.78f};

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
        auto foil = AircraftPart(part, side, controls_);
        Foil(foil, color, skin);
        if (foil.hinged)
          Knuckle(foil, color);
      }
      Nacelle(side);
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

  Vertex BodyVertex(float x, float phi) const {
    Vec3 p = FuselagePoint(x, phi);
    Vec3 dx =
        Add(FuselagePoint(std::min(3.0f, x + 0.001f), phi), Scale(FuselagePoint(std::max(-3.0f, x - 0.001f), phi), -1));
    Vec3 dp = Add(FuselagePoint(x, phi + 0.001f), Scale(FuselagePoint(x, phi - 0.001f), -1));
    Vec3 n = Unit(Cross(dp, dx));
    if (std::abs(x) >= 3)
      n = {x < 0 ? -1.0f : 1.0f, 0, 0};
    return MakeVertex(p, n, skin);
  }

  void Fuselage() {
    constexpr int rings = 160, slices = 96;
    auto point = [&](int ring, int segment) {
      float x = -3 + 6.0f * ring / rings, phi = 2 * pi * segment / slices;
      return BodyVertex(x, phi);
    };
    for (int i = 0; i < rings; ++i)
      for (int j = 0; j < slices; ++j)
        Quad(point(i, j), point(i + 1, j), point(i + 1, (j + 1) % slices), point(i, (j + 1) % slices));
  }

  void Nacelle(int side) {
    // Outer skin and duct wall of the flow-through nacelle. The elliptical lip
    // closes the inlet; an annular face closes the thin exit trailing edge.
    constexpr int rings = 48, slices = 64;
    auto station = [](int i) { return 0.5f * (1 - cosf(pi * i / rings)); };
    auto point = [&](float t, float phi, int surface) {
      Vec3 p = NacellePoint(t, phi, side, surface);
      float a = std::max(0.0f, t - 0.0005f), b = std::min(1.0f, t + 0.0005f);
      float slope = (NacelleRadius(b, surface) - NacelleRadius(a, surface)) / (nacelle_length * (b - a));
      Vec3 radial{0, cosf(phi), sinf(phi)};
      Vec3 n = Scale(Add(Scale(Vec3{1, 0, 0}, -slope), radial), float(surface));
      if (t <= 0)
        n = {-1, 0, 0};
      return MakeVertex(p, n, skin);
    };
    for (int surface : {-1, 1})
      for (int i = 0; i < rings; ++i)
        for (int j = 0; j < slices; ++j) {
          float a = 2 * pi * j / slices, b = 2 * pi * (j + 1) / slices;
          Quad(point(station(i), a, surface), point(station(i + 1), a, surface), point(station(i + 1), b, surface),
               point(station(i), b, surface));
        }
    Vec3 back{1, 0, 0};
    for (int j = 0; j < slices; ++j) {
      float a = 2 * pi * j / slices, b = 2 * pi * (j + 1) / slices;
      Quad(MakeVertex(NacellePoint(1, a, side, 1), back, skin), MakeVertex(NacellePoint(1, b, side, 1), back, skin),
           MakeVertex(NacellePoint(1, b, side, -1), back, skin), MakeVertex(NacellePoint(1, a, side, -1), back, skin));
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

  // Hinge fairing matching HingeKnuckle: a stadium cross-section (the circle of
  // the hinge half-thickness, swept along any Fowler travel) extruded along the
  // hinge line between the control's span limits, with capped ends.
  void Knuckle(const FoilPart &part, Vec3 color) {
    const auto &s = part.spec;
    float slope = (s.tip - s.root) / s.span;
    Vec3 hinge{s.leading + 0.72f * s.root, FoilCamber(0.72f, s.root, s.camber), 0};
    Vec3 axis = Unit({s.sweep + 0.72f * slope, s.dihedral + FoilCamber(0.72f, slope, s.camber), 1});
    float extension = part.fowler ? part.deflection / 40 : 0;
    Vec3 shift{0.10f * s.root * extension, -0.025f * s.root * extension, 0};
    Vec3 travel = Add(shift, Scale(axis, -Dot(shift, axis)));
    Vec3 e1 = Length(travel) > 1e-6f ? Unit(travel) : Unit(Cross(axis, Vec3{0, 0, 1}));
    if (Length(travel) <= 1e-6f && Length(e1) < 0.5f)
      e1 = Unit(Cross(axis, Vec3{0, 1, 0}));
    Vec3 e2 = Cross(axis, e1);
    constexpr int around = 48, along = 16;
    auto point = [&](float z, int k, Vec3 &normal) {
      float angle = 2 * pi * k / around;
      Vec3 d = Add(Scale(e1, cosf(angle)), Scale(e2, sinf(angle)));
      Vec3 center = Add(hinge, Scale(axis, z / axis.z));
      if (cosf(angle) >= 0)
        center = Add(center, travel);
      float chord = s.root + (s.tip - s.root) * std::min(std::max(z / s.span, 0.0f), 1.0f);
      normal = FoilVector(s, d);
      return Add(s.origin, FoilVector(s, Add(center, Scale(d, FoilThickness(0.72f, chord, s.thickness)))));
    };
    for (int j = 0; j < along; ++j) {
      float a = part.begin + (part.end - part.begin) * j / along,
            b = part.begin + (part.end - part.begin) * (j + 1) / along;
      for (int k = 0; k < around; ++k) {
        Vec3 n[4];
        Vec3 p[4] = {point(a, k, n[0]), point(b, k, n[1]), point(b, k + 1, n[2]), point(a, k + 1, n[3])};
        Quad(MakeVertex(p[0], n[0], color), MakeVertex(p[1], n[1], color), MakeVertex(p[2], n[2], color),
             MakeVertex(p[3], n[3], color));
      }
    }
    for (float z : {part.begin, part.end}) {
      Vec3 normal = FoilVector(s, Scale(Vec3{0, 0, 1}, z == part.begin ? -1.0f : 1.0f)), unused;
      Vec3 middle = Add(s.origin, FoilVector(s, Add(Add(hinge, Scale(axis, z / axis.z)), Scale(travel, 0.5f))));
      for (int k = 0; k < around; ++k)
        Triangle(MakeVertex(middle, normal, color), MakeVertex(point(z, k, unused), normal, color),
                 MakeVertex(point(z, k + 1, unused), normal, color));
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
