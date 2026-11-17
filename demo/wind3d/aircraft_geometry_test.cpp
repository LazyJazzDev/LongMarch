#include <algorithm>
#include <cmath>
#include <iostream>

#include "simulation.h"

int main() {
  using namespace wind3d;
  Controls controls;
  auto wing = AircraftPart(0, 1, controls);
  auto pylon = AircraftPart(6, 1, controls);
  if (std::abs(wing.spec.camber - .04f) > 1e-7f || std::abs(pylon.spec.span - .18f) > 1e-7f ||
      std::abs(pylon.spec.origin.y + .26f) > 1e-7f || NacelleRadius(.2f, 1) < .24f)
    return 1;
  size_t checked = 0;
  for (float deflection : {-20.f, -10.f, 0.f, 10.f, 20.f, 40.f}) {
    controls.flap = std::max(0.f, deflection);
    controls.aileron = controls.elevator = controls.rudder = std::clamp(deflection, -20.f, 20.f);
    controls.angle = 5;
    for (int side : {-1, 1})
      for (int part : {1, 2, 4, 5}) {
        auto foil = AircraftPart(part, side, controls);
        for (int station = 1; station < 10; ++station) {
          float span = foil.begin + (foil.end - foil.begin) * station / 10;
          // The line between the fixed and deflected hinge sections must stay
          // inside the fairing, including the full Fowler translation.
          for (float surface : {-.8f, 0.f, .8f}) {
            auto fixed = FoilPoint(foil, .72f, span, 0, false);
            float chord = foil.spec.root + (foil.spec.tip - foil.spec.root) * span / foil.spec.span;
            auto radial = Scale(foil.spec.y_axis, surface * FoilThickness(.72f, chord, foil.spec.thickness));
            fixed = Add(fixed, radial);
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
      if (std::abs(Length(vertex.normal) - 1) > 2e-5f || AircraftDistance(vertex.position, controls) > .003f)
        return 5;
      bool color_ok = false;
      for (Vec3 color : {Vec3{.78f, .78f, .78f}, Vec3{.72f, .66f, .52f}, Vec3{.45f, .61f, .72f}, Vec3{.48f, .65f, .59f},
                         Vec3{.35f, .49f, .69f}})
        color_ok |= Length(Add(vertex.color, Scale(color, -1))) < 1e-6f;
      if (!color_ok)
        return 6;
    }
  }
  std::cout << "PASS " << checked
            << " sealed hinge samples, mesh/occupancy agreement, neutral skin and retained nacelle/pylon\n";
}
