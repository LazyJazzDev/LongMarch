#pragma once

#include <cmath>

#ifdef __CUDACC__
#define WIND3D_HD __host__ __device__ inline
#else
#define WIND3D_HD inline
#endif

namespace wind3d {

struct Vec3 {
  float x{}, y{}, z{};
};

struct Airframe {
  float wing_span{2.65f};  // Half-span excluding the selected tip device.
  float wing_camber{0.025f};
  float wing_sweep{0.55f};  // Leading-edge x offset per metre of span.
  float tail_span{0.98f};
  float tail_leading{1.65f};
  float tail_incidence{};  // Positive raises the stabilizer leading edge.
  bool winglets{true};
};

WIND3D_HD bool ValidAirframe(const Airframe &a) {
  return a.wing_span >= 2 && a.wing_span <= 4.2f && a.wing_camber >= 0 && a.wing_camber <= .06f && a.wing_sweep >= 0 &&
         a.wing_sweep <= .7f && a.tail_span >= .6f && a.tail_span <= 1.4f && a.tail_leading >= 1 &&
         a.tail_leading <= 2 && a.tail_incidence >= -10 && a.tail_incidence <= 10;
}

WIND3D_HD bool SameAirframe(const Airframe &a, const Airframe &b) {
  return a.wing_span == b.wing_span && a.wing_camber == b.wing_camber && a.wing_sweep == b.wing_sweep &&
         a.tail_span == b.tail_span && a.tail_leading == b.tail_leading && a.tail_incidence == b.tail_incidence &&
         a.winglets == b.winglets;
}

struct WingControls {
  bool linked_flaps{true};
  float outer_flap{};
  float cruise_camber{};  // Symmetric trailing-edge bias, degrees.
  float aileron_droop{};  // Symmetric bias, independent of roll command.
};

WIND3D_HD bool ValidWingControls(const WingControls &w) {
  return w.outer_flap >= 0 && w.outer_flap <= 40 && w.cruise_camber >= -5 && w.cruise_camber <= 5 &&
         w.aileron_droop >= 0 && w.aileron_droop <= 15;
}

WIND3D_HD bool SameWingControls(const WingControls &a, const WingControls &b) {
  return a.linked_flaps == b.linked_flaps && a.outer_flap == b.outer_flap && a.cruise_camber == b.cruise_camber &&
         a.aileron_droop == b.aileron_droop;
}

struct Controls {
  float speed{12};
  float angle{5};
  float flap{};
  float aileron{};
  float elevator{};
  float rudder{};
  Airframe airframe;
  WingControls wing_controls;
};

WIND3D_HD bool SameAircraftGeometry(const Controls &a, const Controls &b) {
  return a.angle == b.angle && a.flap == b.flap && a.aileron == b.aileron && a.elevator == b.elevator &&
         a.rudder == b.rudder && SameAirframe(a.airframe, b.airframe) &&
         SameWingControls(a.wing_controls, b.wing_controls);
}

constexpr int aircraft_part_count = 8;

WIND3D_HD Vec3 Add(Vec3 a, Vec3 b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

WIND3D_HD Vec3 Scale(Vec3 a, float s) {
  return {a.x * s, a.y * s, a.z * s};
}

WIND3D_HD Vec3 Cross(Vec3 a, Vec3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

WIND3D_HD float Length(Vec3 a) {
  return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}

WIND3D_HD Vec3 RotateZ(Vec3 p, float degrees) {
  float a = degrees * 0.01745329252f;
  return {cosf(a) * p.x - sinf(a) * p.y, sinf(a) * p.x + cosf(a) * p.y, p.z};
}

// World coordinates: +x downstream, +y up, +z starboard. Positive incidence
// raises the nose (-x). All angles are degrees. Moments use the right-hand rule.
WIND3D_HD Vec3 BodyPoint(Vec3 world, const Controls &controls) {
  return RotateZ(world, controls.angle);
}

WIND3D_HD float Dot(Vec3 a, Vec3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

WIND3D_HD Vec3 Unit(Vec3 p) {
  return Scale(p, 1 / fmaxf(Length(p), 1e-12f));
}

WIND3D_HD float Smooth(float t) {
  t = fminf(fmaxf(t, 0), 1);
  return t * t * (3 - 2 * t);
}

// A constant-diameter cabin, rounded nose and upswept tailcone. The same
// longitudinal and angular profiles define the explicit skin and solid test.
WIND3D_HD float FuselageCenter(float x) {
  return -0.045f * (1 - Smooth(x + 3)) + 0.12f * Smooth((x - 1.5f) / 1.5f);
}

WIND3D_HD float FuselageRadius(float x, float cosine) {
  float radius = 0.315f;
  if (x < -2)
    radius *= sqrtf(fmaxf(0, 1 - (x + 2) * (x + 2)));
  if (x > 1.5f)
    radius *= 1 - Smooth((x - 1.5f) / 1.5f);
  // Longitudinal belly canoe and bilateral wing-root shoulders. Compact
  // quartic envelopes have zero slope at their ends, avoiding hard ridges.
  float u = (x + 0.15f) / 1.3f;
  float envelope = fmaxf(0, 1 - u * u);
  envelope *= envelope;
  float shoulder = (cosine + 0.48f) / 0.48f;
  shoulder = fmaxf(0, 1 - shoulder * shoulder);
  float belly = 0.045f * (1 - cosine) * .5f;
  return radius + envelope * (belly + 0.075f * shoulder * shoulder);
}

WIND3D_HD Vec3 FuselagePoint(float x, float phi) {
  float radius = FuselageRadius(x, cosf(phi));
  return {x, FuselageCenter(x) + radius * cosf(phi), radius * sinf(phi)};
}

WIND3D_HD float FoilThickness(float u, float chord, float thickness = 0.12f) {
  if (u <= 0 || u >= 1)
    return 0;
  return 5 * thickness * chord *
         (0.2969f * sqrtf(u) - 0.126f * u - 0.3516f * u * u + 0.2843f * u * u * u - 0.1036f * u * u * u * u);
}

WIND3D_HD float FoilCamber(float u, float chord, float camber) {
  constexpr float peak = 0.4f;
  return u < peak ? camber * chord / (peak * peak) * (2 * peak * u - u * u)
                  : camber * chord / ((1 - peak) * (1 - peak)) * (1 - 2 * peak + 2 * peak * u - u * u);
}

struct FoilSpec {
  Vec3 origin, x_axis, y_axis, z_axis;
  float span, root, tip, leading, sweep, dihedral, camber, thickness;
  float washout{}, tip_camber_scale{1}, tip_thickness_scale{1};
  bool blended_tip{};
};

// Circular root blend followed by a straight canted tip. Cross-sections rotate
// with the centerline, preserving true normal thickness. Both maps are analytic.
WIND3D_HD Vec3 TipBend(const FoilSpec &s, Vec3 p, bool inverse = false) {
  if (!s.blended_tip || p.z < 0)
    return p;
  constexpr float radius = .18f, angle = 1.221730476f;  // 70 degrees.
  float sa = sinf(angle), ca = cosf(angle), cy = radius * (1 - ca), cz = radius * sa;
  if (inverse) {
    float theta = atan2f(p.z, radius - p.y);
    if (theta <= angle) {
      float y = radius - sqrtf((radius - p.y) * (radius - p.y) + p.z * p.z);
      return {p.x, y, radius * theta};
    }
    float dy = p.y - cy, dz = p.z - cz;
    return {p.x, dy * ca - dz * sa, radius * angle + dy * sa + dz * ca};
  }
  float theta = fminf(p.z / radius, angle), rest = fmaxf(0, p.z - radius * angle);
  return {p.x, radius * (1 - cosf(theta)) + rest * sa + p.y * cosf(theta),
          radius * sinf(theta) + rest * ca - p.y * sinf(theta)};
}

// Reversible section shaping preserves span station. The same deformation
// maps the explicit surface and its analytic solid boundary, including hinges.
WIND3D_HD Vec3 SectionShape(const FoilSpec &s, Vec3 p, bool inverse = false) {
  if (inverse)
    p = TipBend(s, p, true);
  float t = fminf(fmaxf(p.z / s.span, 0), 1);
  float chord = s.root + (s.tip - s.root) * t;
  float leading = s.leading + s.sweep * p.z;
  float camber = s.camber * (1 + (s.tip_camber_scale - 1) * t);
  float thickness_scale = 1 + (s.tip_thickness_scale - 1) * t;
  Vec3 pivot{leading + .25f * chord, s.dihedral * p.z + FoilCamber(.25f, chord, camber), p.z};
  if (inverse)
    p = Add(pivot, RotateZ(Add(p, Scale(pivot, -1)), -s.washout * t));
  float u = fminf(fmaxf((p.x - leading) / chord, 0), 1);
  float base = s.dihedral * p.z + FoilCamber(u, chord, s.camber);
  float shaped = s.dihedral * p.z + FoilCamber(u, chord, camber);
  p.y = inverse ? base + (p.y - shaped) / thickness_scale : shaped + (p.y - base) * thickness_scale;
  return inverse ? p : TipBend(s, Add(pivot, RotateZ(Add(p, Scale(pivot, -1)), s.washout * t)));
}

struct FoilPart {
  FoilSpec spec;
  float begin, end, deflection;
  bool hinged, fowler;
  float deployment{};  // Fowler travel command, separate from cruise camber.
};

WIND3D_HD float AircraftSpanBound(const Controls &controls) {
  return controls.airframe.wing_span + 0.7f;
}

// Parts: buried wing root, inner flap, aileron, tip, horizontal stabilizer,
// vertical stabilizer, nacelle pylon, outer flap. Mirrors share exactly one specification.
WIND3D_HD FoilPart AircraftPart(int part, int side, const Controls &controls) {
  const auto &airframe = controls.airframe;
  Vec3 ex{1, 0, 0}, ey{0, 1, 0}, ez{0, 0, float(side)};
  FoilSpec wing{{0, -0.18f, 0},
                RotateZ(ex, -1.5f),
                RotateZ(ey, -1.5f),
                ez,
                airframe.wing_span,
                0.96f,
                0.22f,
                -1.04f,
                airframe.wing_sweep,
                0.09f,
                airframe.wing_camber,
                0.15f,
                3.0f,
                0.6f,
                0.8f};
  if (part == 0)
    return {wing, 0, 0.35f, 0, false, false};
  float flap_split = airframe.wing_span * .36f;
  float flap_end = airframe.wing_span * (1.75f / 2.7f);
  if (part == 1 || part == 7) {
    float deployment =
        part == 1 || controls.wing_controls.linked_flaps ? controls.flap : controls.wing_controls.outer_flap;
    return {wing,
            part == 1 ? .35f : flap_split,
            part == 1 ? flap_split : flap_end,
            deployment + controls.wing_controls.cruise_camber,
            true,
            true,
            deployment};
  }
  if (part == 2)
    return {wing,
            flap_end,
            airframe.wing_span,
            side * controls.aileron + controls.wing_controls.cruise_camber + controls.wing_controls.aileron_droop,
            true,
            false};
  if (part == 3) {
    float leading = wing.leading + wing.sweep * wing.span;
    Vec3 root =
        SectionShape(wing, {leading, wing.dihedral * wing.span + FoilCamber(0, wing.tip, wing.camber), wing.span});
    wing.origin =
        Add(wing.origin, Add(Add(Scale(wing.x_axis, root.x), Scale(wing.y_axis, root.y)), Scale(wing.z_axis, root.z)));
    wing.x_axis = RotateZ(wing.x_axis, wing.washout);
    wing.y_axis = RotateZ(wing.y_axis, wing.washout);
    wing.camber *= wing.tip_camber_scale;
    wing.thickness *= wing.tip_thickness_scale;
    wing.washout = 0;
    wing.tip_camber_scale = wing.tip_thickness_scale = 1;
    wing.leading = 0;
    wing.root = wing.tip;
    wing.tip = 0.06f;
    wing.blended_tip = airframe.winglets;
    wing.span = airframe.winglets ? .45f : .21f;
    wing.sweep = airframe.winglets ? .65f : 1.15f;
    wing.dihedral = airframe.winglets ? 0.f : .12f;
    return {wing, 0, wing.span, 0, false, false};
  }
  if (part == 4) {
    // Trim about the quarter-chord of the root, independently of elevator.
    float pivot = airframe.tail_leading + .25f * .62f;
    Vec3 tx = RotateZ(ex, -airframe.tail_incidence), ty = RotateZ(ey, -airframe.tail_incidence);
    Vec3 origin = Add(Vec3{pivot, .12f, 0}, Scale(tx, -pivot));
    return {{origin, tx, ty, ez, airframe.tail_span, .62f, .22f, airframe.tail_leading, .36f, .04f, 0, .12f},
            0,
            airframe.tail_span,
            controls.elevator,
            true,
            false};
  }
  if (part == 5)
    return {{{0, 0.15f, 0}, ex, {0, 0, 1}, {0, 1, 0}, 0.85f, 0.90f, 0.30f, 1.40f, 0.6f, 0, 0, 0.12f},
            0,
            0.85f,
            controls.rudder,
            true,
            false};
  // The pylon root sits inside the nacelle's upper wall, clear of the duct.
  return {{{0, -0.25f, side * 0.92f}, ex, {0, 0, 1}, {0, 1, 0}, 0.12f, 0.40f, 0.46f, -0.65f, 0.65f, 0, 0, 0.12f},
          0,
          0.12f,
          0,
          false,
          false};
}

WIND3D_HD float ProjectedWingSpan(const Controls &controls) {
  auto tip = AircraftPart(3, 1, controls).spec;
  return 2 * (controls.airframe.wing_span + TipBend(tip, {0, 0, tip.span}).z);
}

WIND3D_HD Vec3 FoilVector(const FoilSpec &s, Vec3 p) {
  return Add(Add(Scale(s.x_axis, p.x), Scale(s.y_axis, p.y)), Scale(s.z_axis, p.z));
}

WIND3D_HD Vec3 FoilLocal(const FoilSpec &s, Vec3 p) {
  p = Add(p, Scale(s.origin, -1));
  return {Dot(p, s.x_axis), Dot(p, s.y_axis), Dot(p, s.z_axis)};
}

WIND3D_HD Vec3 RotateAxis(Vec3 p, Vec3 axis, float angle) {
  float a = angle * 0.01745329252f;
  return Add(Add(Scale(p, cosf(a)), Scale(Cross(axis, p), sinf(a))), Scale(axis, Dot(axis, p) * (1 - cosf(a))));
}

// Fowler travel is shared by the flap surface and its rounded leading edge.
// The nose moves with the flap; no solid bridge fills the deployed slot.
WIND3D_HD Vec3 FowlerShift(const FoilPart &part) {
  float extension = part.fowler ? fminf(fmaxf(part.deployment / 20, 0), 1) : 0;
  return {.16f * part.spec.root * extension, -.06f * part.spec.root * extension, 0};
}

// Rotate the entire control about its swept, tapered 3D hinge line, rather
// than independently rotating each section about the world span axis.
WIND3D_HD Vec3 ControlTransform(const FoilPart &part, Vec3 p, bool inverse) {
  const auto &s = part.spec;
  float slope = (s.tip - s.root) / s.span;
  Vec3 hinge{s.leading + 0.72f * s.root, FoilCamber(0.72f, s.root, s.camber), 0};
  Vec3 axis = Unit({s.sweep + 0.72f * slope, s.dihedral + FoilCamber(0.72f, slope, s.camber), 1});
  Vec3 shift = FowlerShift(part);
  if (inverse)
    p = Add(p, Scale(shift, -1));
  p = Add(hinge, RotateAxis(Add(p, Scale(hinge, -1)), axis, inverse ? part.deflection : -part.deflection));
  return inverse ? p : Add(p, shift);
}

// Rounded hinge fairing of a control surface: a capsule around the hinge line
// whose radius is the airfoil half-thickness at the hinge. Each flat hinge cut
// is a diameter of it, so at any deflection it closes the wedge between the
// fixed part and the control, as the rounded nose of a control surface in its
// cove does. A Fowler nose translates with the flap, leaving an open slot.
WIND3D_HD float HingeKnuckle(const FoilPart &part, Vec3 local) {
  const auto &s = part.spec;
  float slope = (s.tip - s.root) / s.span;
  Vec3 hinge{s.leading + 0.72f * s.root, FoilCamber(0.72f, s.root, s.camber), 0};
  Vec3 axis = Unit({s.sweep + 0.72f * slope, s.dihedral + FoilCamber(0.72f, slope, s.camber), 1});
  Vec3 shift = FowlerShift(part);
  Vec3 v = Add(Add(local, Scale(shift, -1)), Scale(hinge, -1));
  Vec3 radial = Add(v, Scale(axis, -Dot(v, axis)));
  // Span station of the hinge-line point nearest to `local`: the knuckle ends
  // in planes perpendicular to the swept hinge line.
  float station = Dot(v, axis) * axis.z;
  float chord = s.root + (s.tip - s.root) * fminf(fmaxf(station / s.span, 0), 1);
  float d = Length(radial) - FoilThickness(0.72f, chord, s.thickness);
  return fmaxf(d, fmaxf(part.begin - station, station - part.end));
}

WIND3D_HD Vec3 FoilPoint(const FoilPart &part, float u, float span, int surface, bool control) {
  const auto &s = part.spec;
  float chord = s.root + (s.tip - s.root) * span / s.span;
  Vec3 p{s.leading + s.sweep * span + u * chord,
         s.dihedral * span + FoilCamber(u, chord, s.camber) + surface * FoilThickness(u, chord, s.thickness), span};
  if (control)
    p = ControlTransform(part, p, false);
  return Add(s.origin, FoilVector(s, SectionShape(s, p)));
}

WIND3D_HD float PartDistance(Vec3 body, const FoilPart &part) {
  Vec3 local = SectionShape(part.spec, FoilLocal(part.spec, body), true);
  // Conservative broad phase including the full permitted control sweep.
  float margin = 0.8f * part.spec.root;
  if (local.z < part.begin - margin || local.z > part.end + margin ||
      fabsf(local.y - part.spec.dihedral * local.z) > margin)
    return 1;
  float best = 1e10f;
  for (int piece = 0; piece < (part.hinged ? 2 : 1); ++piece) {
    Vec3 p = piece ? ControlTransform(part, local, true) : local;
    const auto &s = part.spec;
    float chord = s.root + (s.tip - s.root) * fminf(fmaxf(p.z / s.span, 0), 1);
    float x = p.x - s.leading - s.sweep * p.z;
    float u = fminf(fmaxf(x / chord, 0), 1);
    float d = fabsf(p.y - s.dihedral * p.z - FoilCamber(u, chord, s.camber)) - FoilThickness(u, chord, s.thickness);
    d = fmaxf(d, fmaxf(-x, x - chord));
    if (part.hinged)
      d = fmaxf(d, piece ? 0.72f * chord - x : x - 0.72f * chord);
    d = fmaxf(d, fmaxf(part.begin - p.z, p.z - part.end));
    best = fminf(best, d);
  }
  return part.hinged ? fminf(best, HingeKnuckle(part, local)) : best;
}

// Flow-through nacelle, as in an unpowered wind-tunnel model: an annular
// duct whose wall has an elliptical inlet lip and a thin trailing edge.
// Short, large-diameter fan cowl with a rounded inlet and contracted aft body.
// The open duct remains unpowered; no fan thrust is added to the flow model.
constexpr float nacelle_inlet = -0.98f, nacelle_length = 0.70f, nacelle_y = -0.43f;
// A mild contraction for an unpowered through-duct. The previous 0.085 m
// contraction reduced the nominal exit area to roughly 40% of the inlet.
constexpr float nacelle_exit_contraction = 0.020f;

WIND3D_HD float NacelleMean(float t) {
  return 0.170f + 0.012f * sinf(3.14159265359f * t) - nacelle_exit_contraction * Smooth((t - 0.35f) / 0.65f);
}

WIND3D_HD float NacelleHalfThickness(float t) {
  float lip = fminf(fmaxf(t / 0.08f, 0), 1);
  // The last fifth tapers the wall to a thin exit trailing edge.
  float exit = 1 - 0.7f * Smooth((t - 0.8f) / 0.2f);
  return 0.026f * sqrtf(lip * (2 - lip)) * (1 - 0.45f * t) * exit;
}

// surface: +1 outer skin, -1 duct wall.
WIND3D_HD float NacelleRadius(float t, int surface) {
  return NacelleMean(t) + surface * NacelleHalfThickness(t);
}

WIND3D_HD Vec3 NacellePoint(float t, float phi, int side, int surface) {
  float r = NacelleRadius(t, surface);
  return {nacelle_inlet + nacelle_length * t, nacelle_y + r * cosf(phi), side * 0.92f + r * sinf(phi)};
}

// Aircraft components for load breakdowns.
enum Component { kFuselage, kWing, kHorizontalTail, kVerticalTail, kNacelle, kComponents };

// Signed distance to the aircraft (negative inside) and the nearest component.
WIND3D_HD float AircraftDistance(Vec3 world, const Controls &controls, int &component) {
  Vec3 p = BodyPoint(world, controls);
  component = kFuselage;
  if (p.x < -3.1f || p.x > 3.4f || fabsf(p.z) > AircraftSpanBound(controls) || p.y < -0.85f || p.y > 1.4f)
    return 1;
  float dy = p.y - FuselageCenter(p.x), radial = sqrtf(dy * dy + p.z * p.z);
  float body = fmaxf(radial - FuselageRadius(p.x, dy / fmaxf(radial, 1e-12f)), fabsf(p.x) - 3);
  auto keep = [&](float d, int c) {
    if (d < body) {
      body = d;
      component = c;
    }
  };
  for (int side = -1; side <= 1; side += 2) {
    for (int part = 0; part < aircraft_part_count; ++part) {
      if (part == 5 && side == -1)
        continue;
      keep(PartDistance(p, AircraftPart(part, side, controls)), (part <= 3 || part == 7) ? kWing
                                                                : part == 4              ? kHorizontalTail
                                                                : part == 5              ? kVerticalTail
                                                                                         : kNacelle);
    }
    float t = fminf(fmaxf((p.x - nacelle_inlet) / nacelle_length, 0), 1);
    float y = p.y - nacelle_y, z = p.z - side * 0.92f;
    keep(fmaxf(fabsf(sqrtf(y * y + z * z) - NacelleMean(t)) - NacelleHalfThickness(t),
               fmaxf(nacelle_inlet - p.x, p.x - nacelle_inlet - nacelle_length)),
         kNacelle);
  }
  return body;
}

WIND3D_HD float AircraftDistance(Vec3 world, const Controls &controls) {
  int component;
  return AircraftDistance(world, controls, component);
}

}  // namespace wind3d

#undef WIND3D_HD
