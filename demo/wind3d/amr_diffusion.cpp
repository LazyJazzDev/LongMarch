#include <algorithm>
#include <cmath>

#include "amr_transport.h"

namespace wind3d::amr {
bool FaceField::DiffuseAndSponge(double dt,
                                 double viscosity,
                                 double length,
                                 double speed,
                                 std::vector<double> &destination) const {
  if (velocity_.empty() || destination.size() != velocity_.size() || dt < 0 || viscosity < 0 || length < 0 ||
      !std::isfinite(dt) || !std::isfinite(viscosity) || !std::isfinite(length) || !std::isfinite(speed))
    return false;
  for (size_t f = 0; f < velocity_.size(); ++f)
    if (!std::isfinite(destination[f]) || dt * viscosity / mesh_.Faces()[f].area > 1.0 / 6)
      return false;
  auto result = destination;
  for (size_t f = 0; f < result.size(); ++f) {
    auto face = mesh_.Faces()[f];
    if (face.aperture == 0) {
      result[f] = 0;
      continue;
    }
    double h = std::sqrt(face.area), lap = 0;
    for (int direction = 0; direction < 3; ++direction)
      for (int sign : {-1, 1}) {
        auto p = face.centre;
        p[direction] += sign * h;
        for (int a = 0; a < 3; ++a)
          p[a] = std::clamp(p[a], origin_[a] + (a == face.axis ? 0 : .5 * h),
                            origin_[a] + tree_.Roots()[a] * root_width_ - (a == face.axis ? 0 : .5 * h));
        double fraction = 0;
        if (direction != face.axis)
          for (int half : {-1, 1}) {
            auto wall = face.centre;
            wall[direction] += sign * .5 * h;
            wall[face.axis] += half * .5 * h;
            int other = NativeFace(wall, direction);
            if (other >= 0 && mesh_.Faces()[other].aperture == 0)
              fraction += .5;
          }
        lap += (1 - fraction) * (VirtualFace(p, face.axis) - velocity_[f]);
      }
    result[f] += dt * viscosity * lap / (h * h);
    if (length > 0) {
      double column = face.centre[0] - (face.axis == 0 ? 0 : .5 * h);
      double distance = origin_[0] + tree_.Roots()[0] * root_width_ - column;
      if (distance < length && column > origin_[0]) {
        double s = std::clamp(1 - distance / length, 0.0, 1.0), blend = .25 * s * s;
        result[f] += blend * ((face.axis == 0 ? speed : 0) - result[f]);
      }
    }
    if (!std::isfinite(result[f]))
      return false;
  }
  destination = std::move(result);
  return true;
}
}  // namespace wind3d::amr
