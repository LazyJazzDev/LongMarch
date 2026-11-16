#include "amr_transport.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace wind3d::amr {
namespace {
double Cubic(double f0, double f1, double f2, double f3, double t) {
  double delta = f2 - f1, d1 = .5 * (f2 - f0), d2 = .5 * (f3 - f1);
  if (delta == 0 || d1 * delta <= 0)
    d1 = 0;
  if (delta == 0 || d2 * delta <= 0)
    d2 = 0;
  double t2 = t * t, t3 = t2 * t;
  return std::clamp((2 * t3 - 3 * t2 + 1) * f1 + (t3 - 2 * t2 + t) * d1 + (-2 * t3 + 3 * t2) * f2 + (t3 - t2) * d2,
                    std::min(f1, f2), std::max(f1, f2));
}
}  // namespace

FaceField::FaceField(const Mesh &mesh, const Topology &tree, double root_width, std::array<double, 3> origin)
    : mesh_(mesh),
      tree_(tree),
      root_width_(root_width),
      origin_(origin) {
}

int FaceField::CellAt(std::array<double, 3> point) const {
  const int n = mesh_.CellsPerBlock();
  int fine[3];
  for (int a = 0; a < 3; ++a) {
    double q = (point[a] - origin_[a]) / root_width_;
    q = std::clamp(q, 0.0, std::nextafter(double(tree_.Roots()[a]), 0.0));
    fine[a] = int(q * n * (1 << tree_.MaxLevel()));
  }
  int block = tree_.Find(fine[0] / n, fine[1] / n, fine[2] / n);
  if (block < 0)
    return -1;
  auto b = tree_.Leaves()[block];
  int scale = 1 << (tree_.MaxLevel() - b.level);
  int x = fine[0] / scale - b.x * n, y = fine[1] / scale - b.y * n, z = fine[2] / scale - b.z * n;
  return block * n * n * n + (z * n + y) * n + x;
}

bool FaceField::ExtendSlip(std::vector<double> &velocity) const {
  if (velocity.size() != mesh_.Faces().size() || mesh_.Cells().empty() || !(root_width_ > 0))
    return false;
  std::vector<std::array<int, 6>> neighbors(velocity.size());
  std::vector<bool> first(velocity.size());
  for (size_t f = 0; f < velocity.size(); ++f) {
    if (!std::isfinite(velocity[f]))
      return false;
    auto face = mesh_.Faces()[f];
    if (face.aperture > 0)
      continue;
    double h = std::sqrt(face.area);
    for (int k = 0; k < 6; ++k) {
      auto p = face.centre;
      p[k / 2] += (k % 2 ? 1 : -1) * h;
      int other = NativeFace(p, face.axis);
      neighbors[f][k] = other;
      first[f] = first[f] || (other >= 0 && mesh_.Faces()[other].aperture > 0);
    }
  }
  auto extended = velocity;
  for (size_t f = 0; f < extended.size(); ++f)
    if (mesh_.Faces()[f].aperture == 0)
      extended[f] = 0;
  for (int layer = 1; layer <= 2; ++layer)
    for (size_t f = 0; f < velocity.size(); ++f) {
      if (mesh_.Faces()[f].aperture > 0 || (layer == 1 ? !first[f] : first[f]))
        continue;
      double sum = 0;
      int used = 0;
      for (int other : neighbors[f])
        if (other >= 0 &&
            (layer == 1 ? mesh_.Faces()[other].aperture > 0 : mesh_.Faces()[other].aperture == 0 && first[other])) {
          sum += extended[other];
          ++used;
        }
      if (used)
        extended[f] = sum / used;
    }
  velocity = std::move(extended);
  return true;
}

bool FaceField::Update(const std::vector<double> &velocity) {
  if (velocity.size() != mesh_.Faces().size() || mesh_.Cells().empty() || !(root_width_ > 0) ||
      !std::isfinite(root_width_))
    return false;
  const size_t n = mesh_.CellsPerBlock();
  if (!n || tree_.Leaves().size() * n * n * n != mesh_.Cells().size())
    return false;
  for (double p : origin_)
    if (!std::isfinite(p))
      return false;
  for (double u : velocity)
    if (!std::isfinite(u))
      return false;
  std::vector<std::array<double, 3>> centres(mesh_.Cells().size());
  for (size_t cell = 0; cell < centres.size(); ++cell) {
    std::array<double, 3> weight{};
    for (int k = mesh_.Offsets()[cell]; k < mesh_.Offsets()[cell + 1]; ++k) {
      int f = std::abs(mesh_.Incidence()[k]) - 1;
      auto face = mesh_.Faces()[f];
      double area = face.area * face.aperture;
      weight[face.axis] += area;
      centres[cell][face.axis] += area * velocity[f];
    }
    for (int a = 0; a < 3; ++a)
      if (weight[a] > 0)
        centres[cell][a] /= weight[a];
  }
  std::vector<std::array<double, 9>> slopes(centres.size());
  std::vector<std::array<double, 6>> limits(centres.size());
  for (size_t cell = 0; cell < centres.size(); ++cell) {
    for (int a = 0; a < 3; ++a)
      limits[cell][2 * a] = limits[cell][2 * a + 1] = centres[cell][a];
    double matrix[3][6]{};
    for (int k = mesh_.Offsets()[cell]; k < mesh_.Offsets()[cell + 1]; ++k) {
      int f = std::abs(mesh_.Incidence()[k]) - 1;
      auto face = mesh_.Faces()[f];
      int other = face.low == int(cell) ? face.high : face.low;
      if (other < 0 || face.aperture == 0)
        continue;
      double d[3], r2 = 0;
      for (int a = 0; a < 3; ++a) {
        d[a] = mesh_.Cells()[other].centre[a] - mesh_.Cells()[cell].centre[a];
        r2 += d[a] * d[a];
        limits[cell][2 * a] = std::min(limits[cell][2 * a], centres[other][a]);
        limits[cell][2 * a + 1] = std::max(limits[cell][2 * a + 1], centres[other][a]);
      }
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
          matrix[a][b] += d[a] * d[b] / r2;
          matrix[a][3 + b] += d[a] * (centres[other][b] - centres[cell][b]) / r2;
        }
    }
    bool nonsingular = true;
    for (int a = 0; a < 3; ++a) {
      int pivot = a;
      for (int b = a + 1; b < 3; ++b)
        if (std::abs(matrix[b][a]) > std::abs(matrix[pivot][a]))
          pivot = b;
      if (std::abs(matrix[pivot][a]) < 1e-12) {
        nonsingular = false;
        break;
      }
      for (int k = 0; k < 6; ++k)
        std::swap(matrix[a][k], matrix[pivot][k]);
      double scale = matrix[a][a];
      for (int k = 0; k < 6; ++k)
        matrix[a][k] /= scale;
      for (int b = 0; b < 3; ++b)
        if (b != a) {
          double factor = matrix[b][a];
          for (int k = 0; k < 6; ++k)
            matrix[b][k] -= factor * matrix[a][k];
        }
    }
    if (nonsingular)
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
          slopes[cell][3 * b + a] = matrix[a][3 + b];
  }
  velocity_ = velocity;
  centres_ = std::move(centres);
  slopes_ = std::move(slopes);
  limits_ = std::move(limits);
  return true;
}

int FaceField::NativeFace(std::array<double, 3> point, int axis) const {
  int cell = CellAt(point);
  if (cell < 0)
    return -1;
  const auto &c = mesh_.Cells()[cell];
  for (int k = mesh_.Offsets()[cell]; k < mesh_.Offsets()[cell + 1]; ++k) {
    int f = std::abs(mesh_.Incidence()[k]) - 1;
    const auto &face = mesh_.Faces()[f];
    if (face.axis != axis)
      continue;
    double error = 0;
    for (int a = 0; a < 3; ++a)
      error = std::max(error, std::abs(point[a] - face.centre[a]));
    if (error < c.width * 1e-9)
      return f;
  }
  return -1;
}

double FaceField::VirtualFace(std::array<double, 3> point, int axis) const {
  int cell = CellAt(point);
  if (cell < 0)
    return 0;
  int native = NativeFace(point, axis);
  if (native >= 0)
    return velocity_[native];
  const auto &c = mesh_.Cells()[cell];
  double value = centres_[cell][axis];
  for (int a = 0; a < 3; ++a)
    value += slopes_[cell][3 * axis + a] * (point[a] - c.centre[a]);
  return std::clamp(value, limits_[cell][2 * axis], limits_[cell][2 * axis + 1]);
}

double FaceField::Component(std::array<double, 3> point, int axis, bool cubic, double spacing) const {
  for (int a = 0; a < 3; ++a) {
    if (!std::isfinite(point[a]))
      return std::numeric_limits<double>::quiet_NaN();
    point[a] = std::clamp(point[a], origin_[a], origin_[a] + tree_.Roots()[a] * root_width_);
  }

  int cell = CellAt(point);
  if (cell < 0)
    return 0;
  double h = spacing > 0 ? spacing : mesh_.Cells()[cell].width;
  double base[3], fraction[3];
  for (int a = 0; a < 3; ++a) {
    double q = (point[a] - origin_[a]) / h - (a == axis ? 0 : .5);
    base[a] = std::floor(q);
    fraction[a] = q - base[a];
  }
  // Match GPU coordinate rounding: a non-binary spacing can otherwise select
  // the opposite reconstruction cell at a coarse/fine boundary.
  auto corner_point = [&](int x, int y, int z) {
    int corner[3]{x, y, z};
    std::array<double, 3> p{};
    for (int a = 0; a < 3; ++a) {
      p[a] = std::fma(base[a] + corner[a] + (a == axis ? 0 : .5), h, origin_[a]);
      p[a] = std::clamp(p[a], origin_[a] + (a == axis ? 0 : .5 * h),
                        origin_[a] + tree_.Roots()[a] * root_width_ - (a == axis ? 0 : .5 * h));
    }
    return p;
  };
  if (cubic) {
    for (int z = 0; z < 2; ++z)
      for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 2; ++x) {
          int f = NativeFace(corner_point(x, y, z), axis);
          if (f < 0 || mesh_.Faces()[f].aperture != 1)
            cubic = false;
        }
    if (cubic) {
      double planes[4];
      for (int z = 0; z < 4; ++z) {
        double rows[4];
        for (int y = 0; y < 4; ++y) {
          double values[4];
          for (int x = 0; x < 4; ++x)
            values[x] = VirtualFace(corner_point(x - 1, y - 1, z - 1), axis);
          rows[y] = Cubic(values[0], values[1], values[2], values[3], fraction[0]);
        }
        planes[z] = Cubic(rows[0], rows[1], rows[2], rows[3], fraction[1]);
      }
      return Cubic(planes[0], planes[1], planes[2], planes[3], fraction[2]);
    }
  }
  double result = 0, weighted = 0, total = 0;
  for (int z = 0; z < 2; ++z)
    for (int y = 0; y < 2; ++y)
      for (int x = 0; x < 2; ++x) {
        int corner[3]{x, y, z};
        std::array<double, 3> p{};
        double weight = 1;
        for (int a = 0; a < 3; ++a) {
          p[a] = std::fma(base[a] + corner[a] + (a == axis ? 0 : .5), h, origin_[a]);
          weight *= corner[a] ? fraction[a] : 1 - fraction[a];
          double lower = origin_[a] + (a == axis ? 0 : .5 * h),
                 upper = origin_[a] + tree_.Roots()[a] * root_width_ - (a == axis ? 0 : .5 * h);
          p[a] = std::clamp(p[a], lower, upper);
        }
        double value = VirtualFace(p, axis);
        result += weight * value;
        int f = NativeFace(p, axis);
        double aperture = f < 0 ? 1 : mesh_.Faces()[f].aperture;
        weighted += aperture * weight * value;
        total += aperture * weight;
      }
  return total > 1e-3 ? weighted / total : result;
}

std::array<double, 3> FaceField::Sample(std::array<double, 3> point) const {
  if (velocity_.empty())
    return {};
  return {Component(point, 0), Component(point, 1), Component(point, 2)};
}

bool FaceField::Advect(double dt, std::vector<double> &destination, bool cubic) const {
  if (velocity_.empty() || !std::isfinite(dt))
    return false;
  std::vector<double> result(velocity_.size());
  for (size_t i = 0; i < result.size(); ++i) {
    auto face = mesh_.Faces()[i];
    if (face.aperture == 0)
      continue;
    auto point = face.centre, mid = point;
    auto u = Sample(point);
    for (int a = 0; a < 3; ++a)
      mid[a] -= .5 * dt * u[a];
    u = Sample(mid);
    for (int a = 0; a < 3; ++a)
      point[a] -= dt * u[a];
    // Keep the transported face grid fixed as departure crosses a level boundary.
    result[i] = Component(point, face.axis, cubic, std::sqrt(face.area));
    if (!std::isfinite(result[i]))
      return false;
  }
  destination = std::move(result);
  return true;
}
}  // namespace wind3d::amr
