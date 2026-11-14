#include <algorithm>
#include <cmath>

#include "amr_mesh.h"

namespace wind3d::amr {
bool Mesh::Embed(const std::function<double(std::array<double, 3>)> &distance) {
  if (!distance || cells_.empty())
    return false;
  std::vector<bool> solid(cells_.size());
  for (size_t i = 0; i < cells_.size(); ++i) {
    double d = distance(cells_[i].centre);
    if (!std::isfinite(d))
      return false;
    solid[i] = d <= 0;
  }
  auto faces = faces_;
  for (auto &face : faces) {
    bool centre_solid = (face.low >= 0 && solid[face.low]) || (face.high >= 0 && solid[face.high]);
    bool blocked = centre_solid;
    double h = std::sqrt(face.area);
    for (int sample = -3; sample <= 3 && !blocked; ++sample) {
      auto p = face.centre;
      p[face.axis] += sample * h / 8;
      if (face.low >= 0 && face.high >= 0) {
        const auto &a = cells_[face.low].centre, &b = cells_[face.high].centre;
        for (int k = 0; k < 3; ++k)
          p[k] = a[k] + (.5 + sample / 8.0) * (b[k] - a[k]);
      }
      double d = distance(p);
      if (!std::isfinite(d))
        return false;
      blocked = d <= 0;
    }
    double aperture = blocked ? 0 : 1;
    if (centre_solid || !blocked) {
      double d = distance(face.centre);
      if (!std::isfinite(d))
        return false;
      if (std::abs(d) > 2 * h)
        aperture = d > 0 ? 1 : 0;
      else {
        int open = 0, u = (face.axis + 1) % 3, v = (face.axis + 2) % 3;
        for (int i = 0; i < 6; ++i)
          for (int j = 0; j < 6; ++j) {
            auto p = face.centre;
            p[u] += ((i + .5) / 6 - .5) * h;
            p[v] += ((j + .5) / 6 - .5) * h;
            double value = distance(p);
            if (!std::isfinite(value))
              return false;
            open += value > 0;
          }
        aperture = 8 * open < 36 ? 0 : std::nearbyint(255.0 * open / 36) / 255;
      }
    }
    if (aperture < 1 && face.low >= 0 && face.high >= 0 && cells_[face.low].width != cells_[face.high].width)
      return false;
    face.aperture = aperture;
  }
  std::vector<bool> active(cells_.size());
  for (const auto &f : faces)
    if (f.aperture > 0) {
      if (f.low >= 0)
        active[f.low] = true;
      if (f.high >= 0)
        active[f.high] = true;
    }
  faces_ = std::move(faces);
  for (size_t i = 0; i < cells_.size(); ++i)
    cells_[i].active = active[i];
  return true;
}

bool Mesh::PressureLoads(const std::vector<double> &pressure, std::array<double, 6> &loads) const {
  std::array<double, 6> result{};
  if (pressure.size() != cells_.size())
    return false;
  for (double p : pressure)
    if (!std::isfinite(p))
      return false;
  for (size_t cell = 0; cell < cells_.size(); ++cell)
    if (cells_[cell].active) {
      for (int k = offsets_[cell]; k < offsets_[cell + 1]; ++k) {
        int index = incidence_[k], sign = index > 0 ? 1 : -1;
        const auto &f = faces_[std::abs(index) - 1];
        double load = sign * f.area * (1 - f.aperture) * pressure[cell];
        result[f.axis] += load;
        result[3 + (f.axis + 1) % 3] += f.centre[(f.axis + 2) % 3] * load;
        result[3 + (f.axis + 2) % 3] -= f.centre[(f.axis + 1) % 3] * load;
      }
    }
  loads = result;
  return true;
}
}  // namespace wind3d::amr
