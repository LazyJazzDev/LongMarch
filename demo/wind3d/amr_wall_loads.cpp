#include "amr_wall_loads.h"

#include <cmath>

#include "wall_model.h"

namespace wind3d::amr {
bool WallLoads(const Mesh &mesh,
               const std::vector<double> &velocity,
               double density,
               double viscosity,
               std::array<double, 6> &loads) {
  if (velocity.size() != mesh.Faces().size() || !(density > 0) || !(viscosity > 0) || !std::isfinite(density) ||
      !std::isfinite(viscosity))
    return false;
  for (double u : velocity)
    if (!std::isfinite(u))
      return false;
  std::array<double, 6> result{};
  for (size_t i = 0; i < mesh.Cells().size(); ++i) {
    auto cell = mesh.Cells()[i];
    if (!cell.active)
      continue;
    std::array<double, 3> normal{}, u{}, area{};
    for (int k = mesh.Offsets()[i]; k < mesh.Offsets()[i + 1]; ++k) {
      int id = mesh.Incidence()[k], f = std::abs(id) - 1;
      auto face = mesh.Faces()[f];
      normal[face.axis] += (id > 0 ? 1 : -1) * face.area * (1 - face.aperture);
      u[face.axis] += face.area * velocity[f];
      area[face.axis] += face.area;
    }
    double wetted = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
    if (wetted == 0)
      continue;
    double dot = 0;
    for (int a = 0; a < 3; ++a) {
      normal[a] /= wetted;
      u[a] /= area[a];
      dot += normal[a] * u[a];
    }
    double speed2 = 0;
    for (int a = 0; a < 3; ++a) {
      u[a] -= dot * normal[a];
      speed2 += u[a] * u[a];
    }
    double wall = WallEffectiveViscosity(float(viscosity), float(std::sqrt(speed2)), float(.5 * cell.width));
    for (int a = 0; a < 3; ++a) {
      double force = u[a] * 2 * density * wall * wetted / cell.width;
      result[a] += force;
      result[3 + (a + 1) % 3] += cell.centre[(a + 2) % 3] * force;
      result[3 + (a + 2) % 3] -= cell.centre[(a + 1) % 3] * force;
    }
  }
  for (double v : result)
    if (!std::isfinite(v))
      return false;
  loads = result;
  return true;
}
}  // namespace wind3d::amr
