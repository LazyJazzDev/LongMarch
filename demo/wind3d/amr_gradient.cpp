#include "amr_gradient.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace wind3d::amr {
namespace {
using Polynomial = std::array<double, 10>;

Polynomial Basis(std::array<double, 3> p) {
  auto [x, y, z] = p;
  return {1, x, y, z, x * x, y * y, z * z, x * y, x * z, y * z};
}

// Pivoted elimination of the small, scaled normal system. A singular or
// ill-conditioned stencil fails explicitly rather than reverting to two-point
// differences at an interface and silently introducing geometric error.
bool Solve(double matrix[10][10], Polynomial &rhs) {
  double norm = 0;
  for (int i = 0; i < 10; ++i)
    norm = std::max(norm, std::abs(matrix[i][i]));
  for (int i = 0; i < 10; ++i) {
    int pivot = i;
    for (int j = i + 1; j < 10; ++j)
      if (std::abs(matrix[j][i]) > std::abs(matrix[pivot][i]))
        pivot = j;
    if (std::abs(matrix[pivot][i]) < norm * 1e-12)
      return false;
    if (pivot != i) {
      for (int k = 0; k < 10; ++k)
        std::swap(matrix[pivot][k], matrix[i][k]);
      std::swap(rhs[pivot], rhs[i]);
    }
    for (int j = i + 1; j < 10; ++j) {
      double factor = matrix[j][i] / matrix[i][i];
      for (int k = i; k < 10; ++k)
        matrix[j][k] -= factor * matrix[i][k];
      rhs[j] -= factor * rhs[i];
    }
  }
  for (int i = 9; i >= 0; --i) {
    for (int k = i + 1; k < 10; ++k)
      rhs[i] -= matrix[i][k] * rhs[k];
    rhs[i] /= matrix[i][i];
  }
  return true;
}
}  // namespace

bool FaceGradient::Build(const Mesh &mesh) {
  if (mesh.Cells().empty())
    return false;
  std::vector<int> offsets{0}, columns;
  std::vector<double> weights;
  const auto &cells = mesh.Cells();
  const auto &faces = mesh.Faces();
  for (const auto &face : faces) {
    if (face.aperture > 0 && face.low >= 0 && face.high >= 0) {
      if (cells[face.low].width == cells[face.high].width) {
        columns.push_back(face.low);
        weights.push_back(-1 / face.distance);
        columns.push_back(face.high);
        weights.push_back(1 / face.distance);
      } else {
        std::vector<int> stencil{face.low, face.high};
        // Two graph rings provide transverse information on both sides,
        // including interfaces ending at physical-domain boundaries.
        for (int ring = 0; ring < 2; ++ring) {
          auto expanded = stencil;
          for (int cell : stencil)
            for (int k = mesh.Offsets()[cell]; k < mesh.Offsets()[cell + 1]; ++k) {
              const auto &f = faces[std::abs(mesh.Incidence()[k]) - 1];
              int other = f.low == cell ? f.high : f.low;
              if (other >= 0 && f.aperture > 0)
                expanded.push_back(other);
            }
          std::sort(expanded.begin(), expanded.end());
          expanded.erase(std::unique(expanded.begin(), expanded.end()), expanded.end());
          stencil = std::move(expanded);
        }
        double h = std::max(cells[face.low].width, cells[face.high].width), matrix[10][10]{};
        std::vector<Polynomial> basis;
        std::vector<double> fit_weights;
        for (int cell : stencil) {
          std::array<double, 3> p{};
          double r2 = 0;
          for (int axis = 0; axis < 3; ++axis) {
            p[axis] = (cells[cell].centre[axis] - face.centre[axis]) / h;
            r2 += p[axis] * p[axis];
          }
          auto v = Basis(p);
          double w = 1 / ((1 + r2) * (1 + r2));
          basis.push_back(v);
          fit_weights.push_back(w);
          for (int i = 0; i < 10; ++i)
            for (int j = 0; j < 10; ++j)
              matrix[i][j] += w * v[i] * v[j];
        }
        Polynomial rhs{};
        rhs[1 + face.axis] = 1 / h;
        if (!Solve(matrix, rhs))
          return false;
        double sum = 0;
        size_t begin = weights.size();
        for (size_t i = 0; i < stencil.size(); ++i) {
          double coefficient = 0;
          for (int j = 0; j < 10; ++j)
            coefficient += basis[i][j] * rhs[j];
          coefficient *= fit_weights[i];
          columns.push_back(stencil[i]);
          weights.push_back(coefficient);
          sum += coefficient;
        }
        // Preserve the pressure gauge exactly up to the final reduction roundoff.
        weights[begin] -= sum;
      }
    }
    if (weights.size() > size_t(std::numeric_limits<int>::max()))
      return false;
    offsets.push_back(int(weights.size()));
  }
  offsets_ = std::move(offsets);
  columns_ = std::move(columns);
  weights_ = std::move(weights);
  return true;
}
}  // namespace wind3d::amr
