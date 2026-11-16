#include "amr_mass.h"

#include <cmath>
#include <limits>
#include <map>

namespace wind3d::amr {
bool FaceMass::Build(const Mesh &mesh) {
  const auto &cells = mesh.Cells();
  const auto &faces = mesh.Faces();
  if (cells.empty() || faces.empty() || faces.size() > size_t(std::numeric_limits<int>::max()))
    return false;
  std::vector<std::map<int, double>> rows(faces.size());
  for (size_t c = 0; c < cells.size(); ++c) {
    int begin = mesh.Offsets()[c], count = mesh.Offsets()[c + 1] - begin;
    if (count == 6) {
      for (int k = 0; k < count; ++k) {
        int f = std::abs(mesh.Incidence()[begin + k]) - 1;
        rows[f][f] += faces[f].area * cells[c].width * .5 * faces[f].aperture;
      }
      continue;
    }
    // Coarse/fine interfaces must be away from embedded surfaces. A cut
    // irregular cell needs its own consistent volume/surface quadrature.
    std::vector<int> index(count), sign(count);
    std::vector<std::array<double, 3>> r(count);
    std::vector<double> scale(count);
    int normal_count[3]{};
    for (int i = 0; i < count; ++i) {
      int incidence = mesh.Incidence()[begin + i];
      int f = index[i] = std::abs(incidence) - 1;
      sign[i] = incidence > 0 ? 1 : -1;
      if (faces[f].aperture != 1)
        return false;
      ++normal_count[faces[f].axis];
      scale[i] = faces[f].area * cells[c].width * .5;
      for (int a = 0; a < 3; ++a)
        r[i][a] = faces[f].area * (faces[f].centre[a] - cells[c].centre[a]);
    }
    for (int a = 0; a < 3; ++a)
      if (!normal_count[a])
        return false;
    auto q = [&](int i, int j) {
      return double(i == j) - (faces[index[i]].axis == faces[index[j]].axis
                                   ? double(sign[i] * sign[j]) / normal_count[faces[index[i]].axis]
                                   : 0);
    };
    double volume = std::pow(cells[c].width, 3);
    // H = R R^T / V + Q^T diag(area*h/2) Q. N^T R = V I and
    // Q N = 0 give H N = R, while the two complementary terms give SPD.
    for (int i = 0; i < count; ++i)
      for (int j = i; j < count; ++j) {
        double value = 0;
        for (int a = 0; a < 3; ++a)
          value += r[i][a] * r[j][a] / volume;
        for (int k = 0; k < count; ++k)
          value += q(k, i) * scale[k] * q(k, j);
        value *= sign[i] * sign[j];
        rows[index[i]][index[j]] += value;
        if (i != j)
          rows[index[j]][index[i]] += value;
      }
  }
  std::vector<int> offsets{0}, columns;
  std::vector<double> weights, diagonal(faces.size());
  for (size_t i = 0; i < faces.size(); ++i) {
    if (faces[i].aperture == 0)
      rows[i][int(i)] = 1;
    diagonal[i] = rows[i][int(i)];
    if (!(diagonal[i] > 0) || !std::isfinite(diagonal[i]))
      return false;
    for (auto [j, value] : rows[i]) {
      if (!std::isfinite(value))
        return false;
      if (value != 0) {
        columns.push_back(j);
        weights.push_back(value);
      }
    }
    if (weights.size() > size_t(std::numeric_limits<int>::max()))
      return false;
    offsets.push_back(int(weights.size()));
  }
  offsets_ = std::move(offsets);
  columns_ = std::move(columns);
  weights_ = std::move(weights);
  diagonal_ = std::move(diagonal);
  return true;
}
}  // namespace wind3d::amr
