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

// Signed implicit distance to a swept, tapered symmetric NACA 0015 surface.
// It is conservative enough for occupancy/surface extraction, not an exact SDF.
// Deflection is an inverse rotation about the 72%-chord hinge. Positive angles
// move the trailing edge down; the separate leading/trailing pieces meet there.
WIND3D_HD float
FoilDistance(Vec3 p, float span, float root, float tip, float leading, float sweep, float dihedral, float deflection) {
  float s = fabsf(p.z);
  float chord = root + (tip - root) * fminf(s / span, 1.0f);
  float x = p.x - leading - sweep * s;
  float y = p.y - dihedral * s;
  float hinge = 0.72f * chord;
  float radians = deflection * 0.01745329252f;
  float tx = cosf(radians) * (x - hinge) - sinf(radians) * y + hinge;
  float ty = sinf(radians) * (x - hinge) + cosf(radians) * y;
  float best = 1.0e10f;
  for (int piece = 0; piece < 2; ++piece) {
    float px = piece ? tx : x, py = piece ? ty : y;
    float u = fminf(fmaxf(px / chord, 0.0f), 1.0f);
    float thickness =
        0.75f * chord *
        (0.2969f * sqrtf(u) - 0.126f * u - 0.3516f * u * u + 0.2843f * u * u * u - 0.1036f * u * u * u * u);
    float d = fmaxf(fabsf(py) - thickness, fmaxf(-px, px - chord));
    d = fmaxf(d, piece ? hinge - px : px - hinge);
    best = fminf(best, d);
  }
  return fmaxf(best, s - span);
}

WIND3D_HD float AircraftDistance(Vec3 world, const Controls &controls) {
  Vec3 p = BodyPoint(world, controls);
  // Prolate body with a tapered tail. Dimensions are metres (small airliner model).
  float radius = 0.38f * sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x / 9.0f));
  float body = fmaxf(sqrtf(p.y * p.y + p.z * p.z) - radius, fabsf(p.x) - 3.0f);
  float s = fabsf(p.z);
  float deflection = s < 2.7f ? controls.flap : (p.z > 0 ? controls.aileron : -controls.aileron);
  float wing = FoilDistance(p, 4.5f, 1.8f, 0.55f, -1.05f, 0.38f, 0.045f, deflection);
  Vec3 tail{p.x, p.y - 0.18f, p.z};
  float horizontal = FoilDistance(tail, 1.65f, 0.95f, 0.4f, 1.65f, 0.32f, 0.05f, controls.elevator);
  // Rotate the horizontal foil into the vertical plane. Only its upper half is present.
  Vec3 fin{p.x, p.z, p.y - 0.1f};
  float vertical = fmaxf(FoilDistance(fin, 1.55f, 1.25f, 0.45f, 1.35f, 0.45f, 0, controls.rudder), -fin.z);
  return fminf(fminf(body, wing), fminf(horizontal, vertical));
}

}  // namespace wind3d

#undef WIND3D_HD
