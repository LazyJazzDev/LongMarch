#include <algorithm>
#include <cmath>
#include <limits>

#include "amr_transport.h"

namespace wind3d::amr {
bool FaceField::AdvectCorrected(double dt, std::vector<double> &destination, bool cubic) const {
  std::vector<double> forward;
  if (!Advect(dt, forward, cubic))
    return false;
  FaceField transported(mesh_, tree_, root_width_, origin_);
  auto extended = forward;
  if (!transported.ExtendSlip(extended) || !transported.Update(extended))
    return false;
  auto departure = [&](std::array<double, 3> point, double step) {
    auto u = Sample(point), mid = point;
    for (int a = 0; a < 3; ++a)
      mid[a] -= .5 * step * u[a];
    u = Sample(mid);
    for (int a = 0; a < 3; ++a)
      point[a] -= step * u[a];
    return point;
  };
  for (size_t f = 0; f < forward.size(); ++f) {
    const auto &face = mesh_.Faces()[f];
    if (face.aperture == 0)
      continue;
    double reverse = transported.Component(departure(face.centre, -dt), face.axis, cubic);
    if (!std::isfinite(reverse))
      return false;
    double corrected = forward[f] + .5 * (velocity_[f] - reverse);
    auto back = departure(face.centre, dt);
    for (int a = 0; a < 3; ++a) {
      if (!std::isfinite(back[a]))
        return false;
      back[a] = std::clamp(back[a], origin_[a], origin_[a] + tree_.Roots()[a] * root_width_);
    }
    int cell = CellAt(back);
    if (cell < 0)
      return false;
    double h = mesh_.Cells()[cell].width, base[3];
    for (int a = 0; a < 3; ++a)
      base[a] = std::floor((back[a] - origin_[a]) / h - (a == face.axis ? 0 : .5));
    double low = std::numeric_limits<double>::infinity(), high = -low;
    for (int z = 0; z < 2; ++z)
      for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 2; ++x) {
          int corner[3]{x, y, z};
          std::array<double, 3> p{};
          for (int a = 0; a < 3; ++a) {
            p[a] = std::fma(base[a] + corner[a] + (a == face.axis ? 0 : .5), h, origin_[a]);
            p[a] = std::clamp(p[a], origin_[a] + (a == face.axis ? 0 : .5 * h),
                              origin_[a] + tree_.Roots()[a] * root_width_ - (a == face.axis ? 0 : .5 * h));
          }
          double value = VirtualFace(p, face.axis);
          low = std::min(low, value);
          high = std::max(high, value);
        }
    if (corrected >= low && corrected <= high)
      forward[f] = corrected;
    if (!std::isfinite(forward[f]))
      return false;
  }
  destination = std::move(forward);
  return true;
}
}  // namespace wind3d::amr
