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
      for (int part = 0; part < aircraft_part_count; ++part) {
        if (VerticalPart(part) && side == -1)
          continue;
        Vec3 color = (part == 1 || part == 7) ? Vec3{0.72f, 0.66f, 0.52f}
                     : part == 2              ? Vec3{0.45f, 0.61f, 0.72f}
                     : part == 4              ? Vec3{0.48f, 0.65f, 0.59f}
                                              : Vec3{0.35f, 0.49f, 0.69f};
        auto foil = AircraftPart(part, side, controls_);
        Foil(foil, color, skin);
        if (foil.hinged)
          Knuckle(foil, foil.fowler ? color : skin);
      }
      for (int panel = 0; panel < spoiler_panel_count; ++panel) {
        auto spoiler = AircraftSpoiler(panel, side, controls_);
        if (spoiler.deflection > 0)
          Spoiler(spoiler);
      }
      for (int panel = 0; panel < slat_panel_count; ++panel) {
        auto slat = AircraftSlat(panel, side, controls_);
        if (slat.deflection > 0)
          Slat(slat);
      }
      Nacelle(side);
    }
    return std::move(vertices_);
  }

 private:
  Vertex MakeVertex(Vec3 p, Vec3 n, Vec3 color) const {
    return {ConstructionToWorld(p, controls_), RotateZ(Normalize(n), -controls_.angle), color};
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
    auto station = [rings](int i) { return 0.5f * (1 - cosf(pi * i / rings)); };
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

  void Spoiler(const SpoilerPart &part) {
    auto position = [&](Vec3 p) {
      return SpoilerPoint(part, spoiler_first + (spoiler_last - spoiler_first) * p.x,
                          part.begin + (part.end - part.begin) * p.z, 2 * p.y - 1);
    };
    Panel(position, {.45f, .61f, .72f});
  }

  void Slat(const SlatPart &part) {
    Panel([&](Vec3 p) { return SlatPoint(part, p.x, part.begin + (part.end - part.begin) * p.z, 2 * p.y - 1); },
          {.72f, .66f, .52f});
  }

  template <class Position>
  void Panel(const Position &position, Vec3 color) {
    // Six sampled parametric faces, including the thin edges. Orient normals
    // against the derivative along the outward parameter direction.
    for (int face = 0; face < 3; ++face)
      for (int side : {0, 1}) {
        auto parameter = [&](float a, float b, float fixed) -> Vec3 {
          if (face == 0)
            return {fixed, a, b};
          if (face == 1)
            return {a, fixed, b};
          return {a, b, fixed};
        };
        auto vertex = [&](float a, float b) {
          constexpr float e = .0001f;
          Vec3 p = position(parameter(a, b, float(side)));
          Vec3 da =
              Add(position(parameter(a + e, b, float(side))), Scale(position(parameter(a - e, b, float(side))), -1));
          Vec3 db =
              Add(position(parameter(a, b + e, float(side))), Scale(position(parameter(a, b - e, float(side))), -1));
          Vec3 outward = Add(position(parameter(a, b, side + e)), Scale(position(parameter(a, b, side - e)), -1));
          Vec3 n = Cross(da, db);
          if (face == 2) {
            // All cap points share a section plane. Use an interior chord
            // station and the full thickness to avoid cancellation at poles.
            da = Add(position(parameter(.501f, b, float(side))), Scale(position(parameter(.499f, b, float(side))), -1));
            db = Add(position(parameter(.5f, 1, float(side))), Scale(position(parameter(.5f, 0, float(side))), -1));
            n = Cross(da, db);
          }
          if (Dot(n, outward) * (side ? 1 : -1) < 0)
            n = Scale(n, -1);
          return MakeVertex(p, n, color);
        };
        int na = face == 0 ? 1 : 12, nb = face == 2 ? 1 : 8;
        for (int a = 0; a < na; ++a)
          for (int b = 0; b < nb; ++b)
            Quad(vertex(float(a) / na, float(b) / nb), vertex(float(a + 1) / na, float(b) / nb),
                 vertex(float(a + 1) / na, float(b + 1) / nb), vertex(float(a) / na, float(b + 1) / nb));
      }
  }

  void Foil(const FoilPart &part, Vec3 control_color, Vec3 fixed_color) {
    float start = part.begin, end = part.end;
    // Separate closed fixed and hinged solids retain their thin trailing edges
    // at every deflection. Spanwise caps close tips and flap/aileron boundaries.
    constexpr int chord_segments = 48, span_segments = 16;
    for (int piece = 0; piece < (part.hinged ? 2 : 1); ++piece) {
      float first = piece ? part.hinge_fraction : 0, last = piece || !part.hinged ? 1 : part.hinge_fraction;
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

  // Rounded control nose matching HingeKnuckle, with capped ends. Fowler
  // translation moves this nose without filling the open slot behind the wing.
  void Knuckle(const FoilPart &part, Vec3 color) {
    const auto &s = part.spec;
    float slope = (s.tip - s.root) / s.span;
    Vec3 hinge{s.leading + part.hinge_fraction * s.root, FoilCamber(part.hinge_fraction, s.root, s.camber), 0};
    Vec3 axis =
        Unit({s.sweep + part.hinge_fraction * slope, s.dihedral + FoilCamber(part.hinge_fraction, slope, s.camber), 1});
    auto placed = [&](Vec3 p) {
      p = SectionShape(s, p);
      if (part.fowler)
        p = ControlTransform(part, p, false);
      return Add(s.origin, FoilVector(s, p));
    };
    Vec3 e1 = Unit(Cross(axis, Vec3{0, 0, 1}));
    if (Length(e1) < 0.5f)
      e1 = Unit(Cross(axis, Vec3{0, 1, 0}));
    Vec3 e2 = Cross(axis, e1);
    constexpr int around = 48, along = 16;
    auto point = [&](float z, int k, Vec3 &normal) {
      float angle = 2 * pi * k / around;
      Vec3 d = Add(Scale(e1, cosf(angle)), Scale(e2, sinf(angle)));
      Vec3 center = Add(hinge, Scale(axis, z / axis.z));
      float chord = s.root + (s.tip - s.root) * std::min(std::max(z / s.span, 0.0f), 1.0f);
      Vec3 direction =
          part.fowler ? Add(ControlTransform(part, d, false), Scale(ControlTransform(part, {}, false), -1)) : d;
      normal = FoilVector(s, direction);
      return placed(Add(center, Scale(d, FoilThickness(part.hinge_fraction, chord, s.thickness))));
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
      Vec3 middle = placed(Add(hinge, Scale(axis, z / axis.z)));
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
