#include "amr_transfer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <unordered_map>

namespace wind3d::amr {
namespace {
// Axis, normal plane, two tangential lower corners and face side in finest units.
using Key = std::array<int64_t, 5>;

struct KeyHash {
  size_t operator()(const Key &key) const {
    size_t hash = 0;
    for (int64_t value : key)
      hash ^= std::hash<int64_t>{}(value) + size_t(0x9e3779b97f4a7c15ULL) + (hash << 6) + (hash >> 2);
    return hash;
  }
};

struct Sum {
  double area{}, open{}, flux{};
};

// Flux equality alone can hide a changed wall distance and friction law.
// Keep all fluid-adjacent cut faces at identical resolution during regrids.
std::map<std::array<double, 5>, double> WallFaces(const Mesh &mesh) {
  std::map<std::array<double, 5>, double> result;
  for (const auto &f : mesh.Faces())
    if (f.aperture < 1 && ((f.low >= 0 && mesh.Cells()[f.low].active) || (f.high >= 0 && mesh.Cells()[f.high].active)))
      result[{double(f.axis), f.centre[0], f.centre[1], f.centre[2], f.area}] = f.aperture;
  return result;
}

bool Near(double a, double b, double scale) {
  return std::abs(a - b) <= 1e-10 * scale;
}

Key Parent(Key k) {
  k[4] *= 2;
  k[2] = k[2] / k[4] * k[4];
  k[3] = k[3] / k[4] * k[4];
  return k;
}

bool FaceKey(const MeshFace &face, double unit, std::array<double, 3> origin, int64_t max_scale, Key &key) {
  double side = std::sqrt(face.area), scale = side / unit;
  if (!std::isfinite(scale) || scale < .5 || scale > double(max_scale) + .5)
    return false;
  int64_t n = std::llround(scale);
  if (n < 1 || (n & (n - 1)) || !Near(scale, double(n), 1))
    return false;
  int a = face.axis, u = (a + 1) % 3, v = (a + 2) % 3;
  double coordinates[3]{(face.centre[a] - origin[a]) / unit, (face.centre[u] - origin[u] - .5 * side) / unit,
                        (face.centre[v] - origin[v] - .5 * side) / unit};
  key = {a, 0, 0, 0, n};
  for (int j = 0; j < 3; ++j) {
    if (!std::isfinite(coordinates[j]) || coordinates[j] < -.5 || coordinates[j] > double(INT32_MAX))
      return false;
    key[j + 1] = std::llround(coordinates[j]);
    if (!Near(coordinates[j], double(key[j + 1]), 1))
      return false;
  }
  return true;
}
}  // namespace

bool TransferVelocity(const Mesh &source,
                      const Topology &source_tree,
                      const std::vector<double> &velocity,
                      const Mesh &target,
                      const Topology &target_tree,
                      double width,
                      std::array<double, 3> origin,
                      std::vector<double> &destination) {
  if (source.Cells().empty() || target.Cells().empty() || source.CellsPerBlock() != target.CellsPerBlock() ||
      source_tree.Roots() != target_tree.Roots() || !(width > 0) || !std::isfinite(width) ||
      velocity.size() != source.Faces().size())
    return false;
  for (double x : origin)
    if (!std::isfinite(x))
      return false;
  for (double u : velocity)
    if (!std::isfinite(u))
      return false;
  if (WallFaces(source) != WallFaces(target))
    return false;
  const int64_t max_scale = int64_t(1) << std::max(source_tree.MaxLevel(), target_tree.MaxLevel());
  double unit = width / (source.CellsPerBlock() * double(max_scale));
  // Hash lookup changes no arithmetic order: accumulate in source-face order.
  std::unordered_map<Key, size_t, KeyHash> native;
  std::unordered_map<Key, Sum, KeyHash> sums;
  native.reserve(source.Faces().size());
  sums.reserve(source.Faces().size());
  for (size_t i = 0; i < source.Faces().size(); ++i) {
    auto face = source.Faces()[i];
    Key key;
    if (!FaceKey(face, unit, origin, max_scale, key) || !native.emplace(key, i).second)
      return false;
    for (;;) {
      auto &sum = sums[key];
      sum.area += face.area;
      sum.open += face.area * face.aperture;
      sum.flux += face.area * face.aperture * velocity[i];
      if (key[4] == max_scale)
        break;
      key = Parent(key);
    }
  }
  FaceField field(source, source_tree, width, origin);
  auto extended = velocity;
  if (!field.ExtendSlip(extended) || !field.Update(extended))
    return false;
  std::vector<double> result(target.Faces().size());
  std::map<size_t, std::vector<size_t>> split;
  for (size_t i = 0; i < target.Faces().size(); ++i) {
    const auto &face = target.Faces()[i];
    Key key;
    if (!FaceKey(face, unit, origin, max_scale, key))
      return false;
    auto found = sums.find(key);
    if (found != sums.end()) {
      const auto &sum = found->second;
      if (!Near(sum.area, face.area, face.area) || !Near(sum.open, face.area * face.aperture, face.area))
        return false;
      result[i] = sum.open > 0 ? sum.flux / sum.open : 0;
      continue;
    }
    while (key[4] < max_scale) {
      key = Parent(key);
      auto parent = native.find(key);
      if (parent != native.end()) {
        split[parent->second].push_back(i);
        break;
      }
    }
    result[i] = face.aperture > 0 ? field.Sample(face.centre)[face.axis] : 0;
    if (!std::isfinite(result[i]))
      return false;
  }
  for (const auto &[parent, children] : split) {
    const auto &coarse = source.Faces()[parent];
    double area = 0, open = 0, flux = 0, lo = velocity[parent], hi = lo;
    for (size_t i : children) {
      const auto &f = target.Faces()[i];
      area += f.area;
      double weight = f.area * f.aperture;
      open += weight;
      flux += weight * result[i];
      if (weight > 0) {
        lo = std::min(lo, result[i]);
        hi = std::max(hi, result[i]);
      }
    }
    if (!Near(area, coarse.area, coarse.area) || !Near(open, coarse.area * coarse.aperture, coarse.area))
      return false;
    if (open == 0)
      continue;
    double mean = flux / open, theta = 1;
    // Recenter deviations onto the parent's flux; reduce slopes when needed
    // so this conservation correction cannot create a new extremum.
    for (size_t i : children)
      if (target.Faces()[i].aperture > 0) {
        double d = result[i] - mean;
        if (d > 0)
          theta = std::min(theta, (hi - velocity[parent]) / d);
        if (d < 0)
          theta = std::min(theta, (lo - velocity[parent]) / d);
      }
    theta = std::clamp(theta, 0.0, 1.0);
    for (size_t i : children)
      result[i] = target.Faces()[i].aperture > 0 ? velocity[parent] + theta * (result[i] - mean) : 0;
  }
  for (double u : result)
    if (!std::isfinite(u))
      return false;
  destination = std::move(result);
  return true;
}
}  // namespace wind3d::amr
