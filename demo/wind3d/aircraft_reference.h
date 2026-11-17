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
          2 * (s.root * s.root + s.root * s.tip + s.tip * s.tip) / (3 * (s.root + s.tip)),
          float(quarter_x / weight) - ModelDatum().x};
}

// Loads are integrated directly about the fixed model origin. No mass center
// is inferred from geometry; a consumer may shift M by -r x F to its own CG.

}  // namespace wind3d
