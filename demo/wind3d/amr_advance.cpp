#include "amr_advance.h"

#include <cmath>

namespace wind3d::amr {
AdvanceResult AdvanceReference(const Mesh &mesh,
                               const Topology &tree,
                               double root_width,
                               std::array<double, 3> origin,
                               double dt,
                               double speed,
                               std::vector<double> &velocity,
                               std::vector<double> &pressure,
                               bool cubic,
                               double density,
                               double wall_viscosity,
                               double viscosity,
                               double sponge_length,
                               bool corrected_transport) {
  AdvanceResult result;
  if (!(dt > 0) || !std::isfinite(dt) || !(density > 0) || !std::isfinite(density) || !std::isfinite(speed))
    return result;
  FaceField field(mesh, tree, root_width, origin, speed);
  auto extended = velocity;
  if (!field.ExtendSlip(extended) || !field.Update(extended))
    return result;
  std::vector<double> next, potential;
  if (!(corrected_transport ? field.AdvectCorrected(dt, next, cubic) : field.Advect(dt, next, cubic)))
    return result;
  if (!field.DiffuseAndSponge(dt, viscosity, sponge_length, speed, next))
    return result;
  for (size_t f = 0; f < mesh.Faces().size(); ++f) {
    auto face = mesh.Faces()[f];
    if (face.aperture == 0)
      next[f] = 0;
    else if ((face.low < 0 || face.high < 0) && !(face.axis == 0 && face.high < 0))
      next[f] = face.axis == 0 ? speed : 0;
  }
  FaceGradient gradient;
  if (!gradient.Build(mesh))
    return result;
  result.projection = ProjectReference(mesh, gradient, next, potential, 1e-8);
  if (!result.projection.converged)
    return result;
  for (double &p : potential) {
    p *= density / dt;
    if (!std::isfinite(p))
      return result;
  }
  if (!mesh.PressureLoads(potential, result.pressure_loads))
    return result;
  for (double load : result.pressure_loads)
    if (!std::isfinite(load))
      return result;
  if (!WallLoads(mesh, extended, density, wall_viscosity, result.friction_loads))
    return result;
  for (int i = 0; i < 6; ++i)
    result.total_loads[i] = result.pressure_loads[i] + result.friction_loads[i];
  velocity = std::move(next);
  pressure = std::move(potential);
  result.advanced = true;
  return result;
}
}  // namespace wind3d::amr
