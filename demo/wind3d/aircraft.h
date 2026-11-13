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

struct Controls {
  float speed{12};
  float angle{5};
  float flap{};
  float aileron{};
  float elevator{};
  float rudder{};
};

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
  float radius = 0.30f;
  if (x < -2)
    radius *= sqrtf(fmaxf(0, 1 - (x + 2) * (x + 2)));
  if (x > 1.5f)
    radius *= 1 - Smooth((x - 1.5f) / 1.5f);
  float u = (x + 0.2f) / 1.1f;
  float fairing = fmaxf(0, 1 - u * u);
  return radius + 0.055f * fairing * fairing * (1 - cosine) * 0.5f;
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
};

struct FoilPart {
  FoilSpec spec;
  float begin, end, deflection;
  bool hinged, fowler;
};

// Parts: buried wing root, flap, aileron, raked tip, horizontal stabilizer,
// vertical stabilizer, nacelle pylon. Mirrors share exactly one specification.
WIND3D_HD FoilPart AircraftPart(int part, int side, const Controls &controls) {
  Vec3 ex{1, 0, 0}, ey{0, 1, 0}, ez{0, 0, float(side)};
  FoilSpec wing{
      {0, -0.18f, 0}, RotateZ(ex, -1.5f), RotateZ(ey, -1.5f), ez, 2.7f, 1.35f, 0.45f, -1.1f, 0.55f, 0.09f, 0.015f,
      0.13f};
  if (part == 0)
    return {wing, 0, 0.35f, 0, false, false};
  if (part == 1)
    return {wing, 0.35f, 1.75f, controls.flap, true, true};
  if (part == 2)
    return {wing, 1.75f, 2.7f, side * controls.aileron, true, false};
  if (part == 3) {
    wing.origin = Add(wing.origin, Add(Scale(wing.y_axis, wing.dihedral * wing.span), Scale(wing.z_axis, wing.span)));
    wing.leading += wing.sweep * wing.span;
    wing.root = wing.tip;
    wing.tip = 0.07f;
    wing.span = 0.32f;
    wing.sweep = 1.15f;
    wing.dihedral = 0.12f;
    return {wing, 0, wing.span, 0, false, false};
  }
  if (part == 4)
    return {{{0, 0.12f, 0}, ex, ey, ez, 1.05f, 0.75f, 0.28f, 1.65f, 0.36f, 0.04f, 0, 0.12f},
            0,
            1.05f,
            controls.elevator,
            true,
            false};
  if (part == 5)
    return {{{0, 0.15f, 0}, ex, {0, 0, 1}, {0, 1, 0}, 1.05f, 1.15f, 0.4f, 1.3f, 0.6f, 0, 0, 0.12f},
            0,
            1.05f,
            controls.rudder,
            true,
            false};
  return {{{0, -0.47f, side * 0.92f}, ex, {0, 0, 1}, {0, 1, 0}, 0.40f, 0.48f, 0.60f, -0.85f, 0.4f, 0, 0, 0.18f},
          0,
          0.40f,
          0,
          false,
          false};
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

// Rotate the entire control about its swept, tapered 3D hinge line, rather
// than independently rotating each section about the world span axis.
WIND3D_HD Vec3 ControlTransform(const FoilPart &part, Vec3 p, bool inverse) {
  const auto &s = part.spec;
  float slope = (s.tip - s.root) / s.span;
  Vec3 hinge{s.leading + 0.72f * s.root, FoilCamber(0.72f, s.root, s.camber), 0};
  Vec3 axis = Unit({s.sweep + 0.72f * slope, s.dihedral + FoilCamber(0.72f, slope, s.camber), 1});
  float extension = part.fowler ? part.deflection / 40 : 0;
  Vec3 shift{0.10f * s.root * extension, -0.025f * s.root * extension, 0};
  if (inverse)
    p = Add(p, Scale(shift, -1));
  p = Add(hinge, RotateAxis(Add(p, Scale(hinge, -1)), axis, inverse ? part.deflection : -part.deflection));
  return inverse ? p : Add(p, shift);
}

WIND3D_HD Vec3 FoilPoint(const FoilPart &part, float u, float span, int surface, bool control) {
  const auto &s = part.spec;
  float chord = s.root + (s.tip - s.root) * span / s.span;
  Vec3 p{s.leading + s.sweep * span + u * chord,
         s.dihedral * span + FoilCamber(u, chord, s.camber) + surface * FoilThickness(u, chord, s.thickness), span};
  if (control)
    p = ControlTransform(part, p, false);
  return Add(s.origin, FoilVector(s, p));
}

WIND3D_HD float PartDistance(Vec3 body, const FoilPart &part) {
  Vec3 local = FoilLocal(part.spec, body);
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
  return best;
}

WIND3D_HD float NacelleRadius(float t) {
  return 0.175f + 0.018f * sinf(3.14159265359f * t) - 0.035f * t;
}

WIND3D_HD Vec3 NacellePoint(float t, float phi, int side) {
  float r = NacelleRadius(t);
  return {-1.3f + 0.9f * t, -0.54f + r * cosf(phi), side * 0.92f + r * sinf(phi)};
}

WIND3D_HD float AircraftDistance(Vec3 world, const Controls &controls) {
  Vec3 p = BodyPoint(world, controls);
  if (p.x < -3.1f || p.x > 3.4f || fabsf(p.z) > 3.4f || p.y < -0.85f || p.y > 1.4f)
    return 1;
  float dy = p.y - FuselageCenter(p.x), radial = sqrtf(dy * dy + p.z * p.z);
  float body = fmaxf(radial - FuselageRadius(p.x, dy / fmaxf(radial, 1e-12f)), fabsf(p.x) - 3);
  for (int side = -1; side <= 1; side += 2) {
    for (int part = 0; part < 7; ++part) {
      if (part == 5 && side == -1)
        continue;
      body = fminf(body, PartDistance(p, AircraftPart(part, side, controls)));
    }
    float t = fminf(fmaxf((p.x + 1.3f) / 0.9f, 0), 1);
    float y = p.y + 0.54f, z = p.z - side * 0.92f;
    float nacelle = fmaxf(sqrtf(y * y + z * z) - NacelleRadius(t), fmaxf(-1.3f - p.x, p.x + 0.4f));
    body = fminf(body, nacelle);
  }
  return body;
}

}  // namespace wind3d

#undef WIND3D_HD
