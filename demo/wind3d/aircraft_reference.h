#pragma once

#include "aircraft.h"

namespace wind3d {
// Fixed normalization convention: the main-wing trapezoid, including its
// buried carry-through and excluding tip devices. This keeps coefficients
// comparable when a winglet or high-lift device is selected.
struct AircraftReference {
  float area{}, span{}, chord{}, quarter_chord_x{};
};

inline AircraftReference MainWingReference(const Controls &controls = {}) {
  auto wing = AircraftPart(0, 1, controls);
  const auto &s = wing.spec;
  double weight = 0, quarter_x = 0;
  constexpr int samples = 512;
  for (int i = 0; i < samples; ++i) {
    float t = (i + .5f) / samples;
    float chord = s.root + (s.tip - s.root) * t;
    quarter_x += chord * FoilPoint(wing, .25f, s.span * t, 0, false).x;
    weight += chord;
  }
  return {s.span * (s.root + s.tip), 2 * s.span,
          2 * (s.root * s.root + s.root * s.tip + s.tip * s.tip) / (3 * (s.root + s.tip)), float(quarter_x / weight)};
}

// Assumed trim CG: longitudinal quarter chord, body y=z=0. It is not a
// mass-properties estimate of a production aircraft.
inline Vec3 ReferenceCG(const Controls &controls = {}) {
  return {MainWingReference(controls).quarter_chord_x, 0, 0};
}

inline Vec3 MomentAtReferenceCG(Vec3 force, Vec3 moment, const Controls &controls) {
  Vec3 offset = RotateZ(ReferenceCG(controls), -controls.angle);
  return Add(moment, Scale(Cross(offset, force), -1));
}

}  // namespace wind3d
