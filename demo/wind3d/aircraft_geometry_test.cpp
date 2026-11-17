#include <algorithm>
#include <cmath>
#include <iostream>

#include "aircraft_reference.h"
#include "simulation.h"

namespace {
bool PylonAttachments() {
  using namespace wind3d;
  for (float span : {2.f, 2.65f, 4.2f})
    for (float sweep : {0.f, .55f, .7f})
      for (float camber : {0.f, .025f, .06f})
        for (float incidence : {0.f, 1.5f, 4.f, 6.f})
          for (int side : {-1, 1}) {
            Controls c;
            c.airframe.wing_span = span;
            c.airframe.wing_sweep = sweep;
            c.airframe.wing_camber = camber;
            c.airframe.wing_incidence = incidence;
            c.flap = 40;
            c.aileron = 25;
            auto pylon = AircraftPart(6, side, c);
            if (!(pylon.spec.span > .05f))
              return false;
            for (int i = 0; i <= 100; ++i)
              for (int surface : {-1, 1}) {
                auto top = FoilPoint(pylon, i / 100.f, pylon.end, surface, false);
                float distance = 1;
                for (int part : {0, 1, 2, 3, 7, 12})
                  distance = std::min(distance, PartDistance(top, AircraftPart(part, side, c)));
                if (distance > -.0001f) {
                  std::cerr << "Exposed pylon upper cap: " << distance << '\n';
                  return false;
                }
                auto root = FoilPoint(pylon, i / 100.f, 0, surface, false);
                float t = (root.x - nacelle_inlet) / nacelle_length;
                float radius = std::hypot(root.y - nacelle_y, root.z - side * .92f);
                if (t < 0 || t > 1 || radius < NacelleRadius(t, -1) + .0001f || radius > NacelleRadius(t, 1) - .0001f) {
                  std::cerr << "Pylon lower cap leaves nacelle wall: incidence=" << incidence << " span=" << span
                            << " sweep=" << sweep << " camber=" << camber << " u=" << i / 100.f << " radius=" << radius
                            << " inner=" << NacelleRadius(t, -1) << " outer=" << NacelleRadius(t, 1) << '\n';
                  return false;
                }
              }
          }
  return true;
}

bool RigidControls() {
  using namespace wind3d;
  Controls controls;
  for (int id : {1, 2, 4, 5, 7})
    for (int side : {-1, 1})
      for (float angle : {-20.f, -8.f, 15.f, 30.f}) {
        auto neutral = AircraftPart(id, side, controls), moved = neutral;
        moved.deflection = angle;
        moved.deployment = moved.fowler ? std::max(angle, 0.f) : 0;
        std::vector<Vec3> before, after;
        for (float fraction : {.05f, .3f, .6f, .95f})
          for (float chord_fraction : {.05f, .5f, 1.f})
            for (int surface : {-1, 1}) {
              float span = neutral.begin + (neutral.end - neutral.begin) * fraction;
              float u = neutral.hinge_fraction + (1 - neutral.hinge_fraction) * chord_fraction;
              before.push_back(FoilPoint(neutral, u, span, surface, true));
              after.push_back(FoilPoint(moved, u, span, surface, true));
              auto local = FoilLocal(moved.spec, after.back());
              auto restored = ControlTransform(moved, ControlTransform(moved, local, true), false);
              if (Length(Add(restored, Scale(local, -1))) > 2e-6f)
                return false;
            }
        float maximum = 0;
        for (size_t i = 0; i < before.size(); ++i)
          for (size_t j = i + 1; j < before.size(); ++j) {
            float a = Length(Add(before[i], Scale(before[j], -1)));
            float b = Length(Add(after[i], Scale(after[j], -1)));
            maximum = std::max(maximum, std::abs(a - b));
          }
        if (maximum > 2e-5f) {
          std::cerr << "Non-rigid control: part=" << id << " side=" << side << " angle=" << angle
                    << " distance_change=" << maximum << '\n';
          return false;
        }
      }
  return true;
}
}  // namespace

int main() {
  using namespace wind3d;
  if (!RigidControls())
    return 32;
  if (!PylonAttachments())
    return 35;
  for (float x : {-2.9f, -2.72f, -2.48f, -2.26f, -2.f}) {
    constexpr float h = .0002f;
    float left = (CockpitRoof(x) - CockpitRoof(x - h)) / h;
    float right = (CockpitRoof(x + h) - CockpitRoof(x)) / h;
    if (std::abs(left - right) > .01f)
      return 36;
  }
  for (int i = 1; i < 100; ++i) {
    float x = -3 + i / 100.f;
    auto top = FuselagePoint(x, 0);
    if (std::abs(top.y - CockpitRoof(x)) > 1e-6f || top.y > .315001f)
      return 37;
    for (int j = 0; j < 32; ++j) {
      float radius = FuselageRadius(x, std::cos(j * 6.2831853f / 32));
      if (!(radius > 0) || !std::isfinite(radius))
        return 38;
    }
  }
  for (int part : {8, 9, 10, 11, 12}) {
    Controls neutral, deflected;
    deflected.aileron = deflected.elevator = deflected.rudder = 25;
    auto a = AircraftPart(part, 1, neutral), b = AircraftPart(part, 1, deflected);
    if (a.hinged || b.hinged)
      return 33;
    for (float u : {0.f, .5f, 1.f})
      for (float t : {0.f, .5f, 1.f}) {
        float z = a.begin + (a.end - a.begin) * t;
        if (Length(Add(FoilPoint(a, u, z, 1, false), Scale(FoilPoint(b, u, z, 1, false), -1))) > 1e-7f)
          return 34;
      }
  }
  Controls controls;
  const auto reference = MainWingReference(controls);
  if (std::abs(reference.area - 3.127f) > 1e-5f || std::abs(reference.span - 5.3f) > 1e-5f ||
      reference.quarter_chord_x < -.31f || reference.quarter_chord_x > -.28f)
    return 29;
  Controls other_tip = controls;
  other_tip.airframe.winglets = false;
  if (MainWingReference(other_tip).area != reference.area)
    return 30;
  std::cout << "Main-wing reference: area=" << reference.area << " span=" << reference.span
            << " MAC=" << reference.chord << " quarter_chord_x=" << reference.quarter_chord_x << '\n';
  for (float angle : {0.f, 10.f}) {
    Controls pose;
    pose.angle = angle;
    auto r = RotateZ(ReferenceCG(pose), -angle);
    Vec3 force{0, 100, 0}, applied_at_cg{0, 0, 100 * r.x};
    if (Length(MomentAtReferenceCG(force, applied_at_cg, pose)) > 1e-5f)
      return 31;
  }
  auto wing = AircraftPart(0, 1, controls);
  auto pylon = AircraftPart(6, 1, controls);
  const float length = 6, diameter = 2 * FuselageRadius(-1.5f, 1);
  const float span = ProjectedWingSpan(controls);
  float nacelle_diameter = 0;
  for (int i = 0; i <= 100; ++i)
    nacelle_diameter = std::max(nacelle_diameter, 2 * NacelleRadius(i / 100.f, 1));
  if (span / length < .93f || span / length > .97f || diameter / length < .10f || diameter / length > .11f ||
      nacelle_diameter / diameter < .55f || nacelle_diameter / diameter > .68f || pylon.spec.span <= .05f)
    return 1;
  for (float z : {0.f, .5f, 1.5f, 2.65f}) {
    for (Vec3 point : {Vec3{-.3f, .1f, z}, Vec3{.4f, -.2f, z}}) {
      auto restored = SectionShape(wing.spec, SectionShape(wing.spec, point), true);
      if (Length(Add(restored, Scale(point, -1))) > 2e-6f)
        return 8;
    }
  }
  for (int part : {4, 5, 6})
    for (int side : {-1, 1}) {
      const auto spec = AircraftPart(part, side, controls).spec;
      for (float fraction : {0.f, .1f, .3f, .7f, 1.f})
        for (float u : {0.f, .25f, .72f, 1.f}) {
          float z = spec.span * fraction, chord = spec.root + (spec.tip - spec.root) * fraction;
          Vec3 point{spec.leading + spec.sweep * z + u * chord, .03f, z};
          auto restored = SectionShape(spec, SectionShape(spec, point), true);
          if (Length(Add(restored, Scale(point, -1))) > 3e-6f)
            return 28;
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
  for (float incidence : {0.f, 4.f, 6.f}) {
    Controls installed = controls;
    installed.airframe.wing_incidence = incidence;
    if (!ValidAirframe(installed.airframe) || SameAircraftGeometry(installed, controls))
      return 39;
    auto a = AircraftPart(0, 1, installed).spec, b = AircraftPart(0, 1, controls).spec;
    Vec3 pa = Add(a.origin, Scale(a.x_axis, a.leading + .25f * a.root));
    Vec3 pb = Add(b.origin, Scale(b.x_axis, b.leading + .25f * b.root));
    if (Length(Add(pa, Scale(pb, -1))) > 1e-6f ||
        std::abs(std::atan2(-a.x_axis.y, a.x_axis.x) * 57.2957795f - incidence) > 1e-5f)
      return 40;
  }
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
  for (int field = 0; field < 9; ++field) {
    Controls changed;
    if (field == 0)
      changed.wing_controls.linked_flaps = false;
    if (field == 1)
      changed.wing_controls.outer_flap = 10;
    if (field == 2)
      changed.wing_controls.cruise_camber = 2;
    if (field == 3)
      changed.wing_controls.aileron_droop = 5;
    if (field == 4)
      changed.wing_controls.spoiler = 30;
    if (field == 5)
      changed.wing_controls.spoiler_roll = 10;
    if (field == 6)
      changed.wing_controls.linked_spoiler_roll = true;
    if (field == 7)
      changed.wing_controls.slat = 20;
    if (field == 8)
      changed.wing_controls.linked_slats = true;
    if (SameAircraftGeometry(changed, Controls{}))
      return 19;
  }
  WingControls invalid_slat;
  invalid_slat.slat = 31;
  if (ValidWingControls(invalid_slat))
    return 27;
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
      controls.wing_controls.spoiler = std::clamp(1.5f * deflection, 0.f, 60.f);
      controls.wing_controls.spoiler_roll = deflection < 0 ? -10.f : 10.f;
      controls.wing_controls.linked_spoiler_roll = deflection == 20;
      controls.wing_controls.slat = std::clamp(.75f * deflection, 0.f, 30.f);
      controls.wing_controls.linked_slats = deflection == 20;
      for (int side : {-1, 1})
        for (int panel = 0; panel < slat_panel_count; ++panel) {
          auto slat = AircraftSlat(panel, side, controls);
          float span = .5f * (slat.begin + slat.end);
          for (float u : {0.f, .25f, .5f, .9f, 1.f})
            for (float face : {-1.f, 1.f}) {
              auto p = SlatPoint(slat, u, span, face);
              if (slat.deflection > 0 && std::abs(SlatDistance(p, slat)) > 3e-6f)
                return 23;
              if (slat.deflection == 0 && AircraftDistance(RotateZ(p, -controls.angle), controls) > 3e-6f)
                return 24;
            }
          if (slat.deflection >= 15) {
            // Passage from the slat's aft upper surface to the main wing lower
            // surface at the same developed chord station.
            const auto &s = slat.spec;
            auto a = SlatPoint(slat, .9f, span, 1);
            auto local = SectionShape(s, FoilLocal(s, a), true);
            float chord = s.root + (s.tip - s.root) * span / s.span;
            float u = (local.x - s.leading - s.sweep * span) / chord;
            local.y = s.dihedral * span + FoilCamber(u, chord, s.camber) - FoilThickness(u, chord, s.thickness);
            auto b = Add(s.origin, FoilVector(s, SectionShape(s, local)));
            auto middle = Scale(Add(a, b), .5f);
            if (SlatDistance(middle, slat) <= .001f) {
              std::cerr << "Slat passage: angle=" << slat.deflection << " panel=" << panel << " span=" << span
                        << " camber=" << s.camber << " distance=" << SlatDistance(middle, slat) << '\n';
              return 25;
            }
            if (panel >= 2 && AircraftDistance(RotateZ(middle, -controls.angle), controls) <= .001f)
              return 26;
          }
        }
      for (int side : {-1, 1})
        for (int panel = 0; panel < spoiler_panel_count; ++panel) {
          auto spoiler = AircraftSpoiler(panel, side, controls);
          float span = .5f * (spoiler.begin + spoiler.end);
          for (float u : {spoiler_first, .59f, spoiler_last})
            for (float face : {-1.f, 1.f}) {
              auto p = SpoilerPoint(spoiler, u, span, face);
              if (spoiler.deflection > 0 && std::abs(SpoilerDistance(p, spoiler)) > 3e-6f)
                return 20;
              if (spoiler.deflection == 0 && PartDistance(p, AircraftPart(0, side, controls)) > 0) {
                // Part zero covers only the buried root: query the full wing union.
                if (AircraftDistance(RotateZ(p, -controls.angle), controls) > 3e-6f)
                  return 21;
              }
            }
          if (spoiler.deflection > 10) {
            auto raised = SpoilerPoint(spoiler, spoiler_last, span, 1);
            spoiler.deflection = 0;
            auto stowed = SpoilerPoint(spoiler, spoiler_last, span, 1);
            if (raised.y <= stowed.y + .005f)
              return 22;
          }
        }
      for (int side : {-1, 1})
        for (int part : {1, 2, 4, 5, 7}) {
          auto foil = AircraftPart(part, side, controls);
          for (int station = 1; station < 10; ++station) {
            float span = foil.begin + (foil.end - foil.begin) * station / 10;
            if (foil.fowler && foil.deployment > 0) {
              // A deployed slot must contain fluid between the fixed trailing
              // face and the translated flap nose, on both mirrored wings.
              auto fixed = FoilPoint(foil, foil.hinge_fraction, span, 0, false);
              auto moved = FoilPoint(foil, foil.hinge_fraction, span, 0, true);
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
              auto fixed = FoilPoint(foil, foil.hinge_fraction, span, 0, false);
              auto upper = FoilPoint(foil, foil.hinge_fraction, span, 1, false);
              fixed = Add(fixed, Scale(Add(upper, Scale(fixed, -1)), surface));
              auto local = FoilLocal(foil.spec, fixed);
              auto moved = Add(foil.spec.origin, FoilVector(foil.spec, ControlTransform(foil, local, false)));
              for (int i = 0; i <= 10; ++i) {
                auto p = Add(Scale(fixed, 1 - i / 10.f), Scale(moved, i / 10.f));
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
        if (std::abs(Length(vertex.normal) - 1) > 2e-5f || AircraftDistance(vertex.position, controls) > .003f) {
          std::cerr << "Mesh boundary mismatch: position=" << vertex.position.x << ',' << vertex.position.y << ','
                    << vertex.position.z << " normal=" << Length(vertex.normal)
                    << " distance=" << AircraftDistance(vertex.position, controls) << " deflection=" << deflection
                    << '\n';
          return 5;
        }
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
