#include <algorithm>
#include <cmath>
#include <iostream>

#include "simulation.h"

int main() {
  using namespace wind3d;
  Controls controls;
  auto wing = AircraftPart(0, 1, controls);
  auto pylon = AircraftPart(6, 1, controls);
  const float length = 6, diameter = 2 * FuselageRadius(-1.5f, 1);
  const float span = ProjectedWingSpan(controls);
  float nacelle_diameter = 0;
  for (int i = 0; i <= 100; ++i)
    nacelle_diameter = std::max(nacelle_diameter, 2 * NacelleRadius(i / 100.f, 1));
  if (span / length < .93f || span / length > .97f || diameter / length < .10f || diameter / length > .11f ||
      nacelle_diameter / diameter < .55f || nacelle_diameter / diameter > .68f ||
      std::abs(pylon.spec.span - .12f) > 1e-7f)
    return 1;
  for (float z : {0.f, .5f, 1.5f, 2.65f}) {
    for (Vec3 point : {Vec3{-.3f, .1f, z}, Vec3{.4f, -.2f, z}}) {
      auto restored = SectionShape(wing.spec, SectionShape(wing.spec, point), true);
      if (Length(Add(restored, Scale(point, -1))) > 2e-6f)
        return 8;
    }
  }
  auto tip = AircraftPart(3, 1, controls);
  for (float u : {0.f, .25f, .5f, .9f, 1.f})
    for (int surface : {-1, 1}) {
      auto a = FoilPoint(wing, u, wing.spec.span, surface, false);
      auto b = FoilPoint(tip, u, 0, surface, false);
      if (Length(Add(a, Scale(b, -1))) > 2e-6f)
        return 9;
    }
  Controls trim = controls;
  trim.airframe.tail_incidence = 7;
  if (SameAirframe(trim.airframe, controls.airframe))
    return 10;
  auto trimmed = AircraftPart(4, 1, trim), neutral = AircraftPart(4, 1, controls);
  auto pivot_a = FoilPoint(trimmed, .25f, 0, 0, false), pivot_b = FoilPoint(neutral, .25f, 0, 0, false);
  if (Length(Add(pivot_a, Scale(pivot_b, -1))) > 1e-6f)
    return 11;
  if (FoilPoint(trimmed, 0, 0, 0, false).y <= FoilPoint(neutral, 0, 0, 0, false).y)
    return 12;
  for (float z : {0.f, .05f, .2f, .35f, .45f})
    for (float y : {-.015f, 0.f, .015f}) {
      Vec3 point{.1f, y, z};
      auto restored = TipBend(tip.spec, TipBend(tip.spec, point), true);
      if (Length(Add(restored, Scale(point, -1))) > 2e-6f)
        return 13;
    }
  Controls raked = controls;
  raked.airframe.winglets = false;
  if (SameAirframe(raked.airframe, controls.airframe) || AircraftPart(3, 1, raked).spec.blended_tip)
    return 14;
  Controls split;
  split.flap = 20;
  split.wing_controls = {false, 10, -2, 5};
  if (SameAircraftGeometry(split, Controls{}) || AircraftPart(1, 1, split).deflection != 18 ||
      AircraftPart(7, 1, split).deflection != 8 || AircraftPart(2, 1, split).deflection != 3)
    return 16;
  split.wing_controls.linked_flaps = true;
  if (AircraftPart(7, 1, split).deflection != 18)
    return 17;
  split.flap = 0;
  if (Length(FowlerShift(AircraftPart(1, 1, split))) != 0)
    return 18;
  for (int field = 0; field < 4; ++field) {
    Controls changed;
    if (field == 0)
      changed.wing_controls.linked_flaps = false;
    if (field == 1)
      changed.wing_controls.outer_flap = 10;
    if (field == 2)
      changed.wing_controls.cruise_camber = 2;
    if (field == 3)
      changed.wing_controls.aileron_droop = 5;
    if (SameAircraftGeometry(changed, Controls{}))
      return 19;
  }
  size_t checked = 0, slots = 0;
  for (Airframe airframe : {Airframe{}, Airframe{2.7f, .025f, .55f, 1.05f, 1.65f}, Airframe{2, 0, 0, .6f, 1},
                            Airframe{4.2f, .06f, .7f, 1.4f, 2}, Airframe{2.65f, .025f, .55f, .98f, 1.65f, -10},
                            Airframe{2.65f, .025f, .55f, .98f, 1.65f, 10}}) {
    if (!ValidAirframe(airframe))
      return 7;
    controls.airframe = airframe;
    for (float deflection : {-20.f, -10.f, 0.f, 10.f, 20.f, 40.f}) {
      controls.flap = std::max(0.f, deflection);
      controls.wing_controls = {deflection <= 0, 40 - controls.flap, deflection < 0 ? -5.f : 5.f, 10};
      controls.aileron = controls.elevator = controls.rudder = std::clamp(deflection, -20.f, 20.f);
      controls.angle = 5;
      for (int side : {-1, 1})
        for (int part : {1, 2, 4, 5, 7}) {
          auto foil = AircraftPart(part, side, controls);
          for (int station = 1; station < 10; ++station) {
            float span = foil.begin + (foil.end - foil.begin) * station / 10;
            if (foil.fowler && foil.deployment > 0) {
              // A deployed slot must contain fluid between the fixed trailing
              // face and the translated flap nose, on both mirrored wings.
              auto fixed = FoilPoint(foil, .72f, span, 0, false);
              auto moved = FoilPoint(foil, .72f, span, 0, true);
              bool gap = false;
              for (int i = 1; i < 20; ++i) {
                auto p = Add(Scale(fixed, 1 - i / 20.f), Scale(moved, i / 20.f));
                gap |= AircraftDistance(RotateZ(p, -controls.angle), controls) > .001f;
              }
              if (!gap) {
                std::cerr << "Closed Fowler slot\n";
                return 15;
              }
              ++slots;
              continue;
            }
            // Ordinary hinges and retracted flaps stay sealed.
            for (float surface : {-.8f, 0.f, .8f}) {
              auto fixed = FoilPoint(foil, .72f, span, 0, false);
              fixed = Add(foil.spec.origin,
                          FoilVector(foil.spec, SectionShape(foil.spec, FoilLocal(foil.spec, fixed), true)));
              float chord = foil.spec.root + (foil.spec.tip - foil.spec.root) * span / foil.spec.span;
              auto radial = Scale(foil.spec.y_axis, surface * FoilThickness(.72f, chord, foil.spec.thickness));
              fixed = Add(fixed, radial);
              auto local = FoilLocal(foil.spec, fixed);
              auto moved = Add(foil.spec.origin, FoilVector(foil.spec, ControlTransform(foil, local, false)));
              for (int i = 0; i <= 10; ++i) {
                auto p = Add(Scale(fixed, 1 - i / 10.f), Scale(moved, i / 10.f));
                p = Add(foil.spec.origin, FoilVector(foil.spec, SectionShape(foil.spec, FoilLocal(foil.spec, p))));
                if (PartDistance(p, foil) > 2e-6f) {
                  std::cerr << "Open hinge: part=" << part << " deflection=" << deflection << '\n';
                  return 2;
                }
                ++checked;
              }
            }
          }
        }
      auto mesh = AircraftMesh(controls);
      if (mesh.empty() || mesh.size() % 3)
        return 3;
      for (const auto &vertex : mesh) {
        for (float v : {vertex.position.x, vertex.position.y, vertex.position.z, vertex.normal.x, vertex.normal.y,
                        vertex.normal.z, vertex.color.x, vertex.color.y, vertex.color.z})
          if (!std::isfinite(v))
            return 4;
        if (std::abs(Length(vertex.normal) - 1) > 2e-5f || AircraftDistance(vertex.position, controls) > .003f)
          return 5;
        bool color_ok = false;
        for (Vec3 color : {Vec3{.78f, .78f, .78f}, Vec3{.72f, .66f, .52f}, Vec3{.45f, .61f, .72f},
                           Vec3{.48f, .65f, .59f}, Vec3{.35f, .49f, .69f}})
          color_ok |= Length(Add(vertex.color, Scale(color, -1))) < 1e-6f;
        if (!color_ok)
          return 6;
      }
    }
  }
  std::cout << "PASS " << checked << " sealed hinge samples, " << slots
            << " open slot sections, mesh/occupancy agreement\n";
}
