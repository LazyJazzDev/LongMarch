#include "amr_refinement.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

namespace wind3d::amr {
namespace {
using Key = std::array<int, 4>;

struct Requests {
  std::array<int, 3> roots;
  double width;
  std::array<double, 3> origin;
  const RefinementOptions &options;
  std::set<Key> split;
  size_t root_count{};

  bool Validate() {
    if (!(width > 0) || !std::isfinite(width) || options.max_level < 0 || options.max_level > 12 ||
        options.buffer_blocks < 0 || options.buffer_blocks > 8 || !(options.coarsen_threshold >= 0) ||
        !(options.refine_threshold > options.coarsen_threshold) || !std::isfinite(options.refine_threshold) ||
        !(options.reference_speed > 0) || !std::isfinite(options.reference_speed) || options.protect_padding < 0 ||
        !std::isfinite(options.protect_padding))
      return false;
    root_count = 1;
    for (int a = 0; a < 3; ++a) {
      if (roots[a] <= 0 || roots[a] > (std::numeric_limits<int>::max() >> options.max_level) ||
          !std::isfinite(origin[a]) || !std::isfinite(origin[a] + roots[a] * width) ||
          size_t(roots[a]) > options.capacity / root_count)
        return false;
      root_count *= roots[a];
    }
    return true;
  }

  bool Box(ProtectedBox box, int target_level, double padding) {
    int lo[3], hi[3];
    for (int a = 0; a < 3; ++a)
      if (!std::isfinite(box.lower[a]) || !std::isfinite(box.upper[a]) || !(box.upper[a] > box.lower[a]))
        return false;
    if (target_level == 0)
      return true;
    int level = target_level - 1;
    double h = width / (1 << level);
    size_t count = 1, limit = (options.capacity - root_count) / 7;
    for (int a = 0; a < 3; ++a) {
      double low = std::max(origin[a], box.lower[a] - padding),
             high = std::min(origin[a] + roots[a] * width, box.upper[a] + padding);
      if (high <= low)
        return true;
      lo[a] = std::max(0, int(std::floor((low - origin[a]) / h)));
      hi[a] = std::min((roots[a] << level) - 1, int(std::ceil((high - origin[a]) / h)) - 1);
      size_t span = size_t(hi[a] - lo[a] + 1);
      if (span > limit / count)
        return false;
      count *= span;
    }
    for (int z = lo[2]; z <= hi[2]; ++z)
      for (int y = lo[1]; y <= hi[1]; ++y)
        for (int x = lo[0]; x <= hi[0]; ++x) {
          Key key{level, x, y, z};
          for (;;) {
            if (!split.insert(key).second)
              break;
            if (split.size() > limit)
              return false;
            if (key[0] == 0)
              break;
            --key[0];
            for (int a = 1; a < 4; ++a)
              key[a] /= 2;
          }
        }
    return true;
  }

  bool Protect() {
    for (const auto &box : options.protected_regions)
      if (!Box(box, options.max_level, options.protect_padding))
        return false;
    return true;
  }

  bool Build(Topology &out) {
    return out.Regrid(roots, options.max_level, options.capacity,
                      [&](Block b) { return split.count({b.level, b.x, b.y, b.z}) > 0; });
  }
};

ProtectedBox Bounds(Block b, double width, std::array<double, 3> origin) {
  ProtectedBox box;
  int p[3]{b.x, b.y, b.z};
  double h = width / (1 << b.level);
  for (int a = 0; a < 3; ++a) {
    box.lower[a] = origin[a] + p[a] * h;
    box.upper[a] = origin[a] + (p[a] + 1) * h;
  }
  return box;
}
}  // namespace

bool ProtectedTopology(std::array<int, 3> roots,
                       double width,
                       std::array<double, 3> origin,
                       const RefinementOptions &options,
                       Topology &destination) {
  Requests requests{roots, width, origin, options, {}, 0};
  Topology next;
  if (!requests.Validate() || !requests.Protect() || !requests.Build(next))
    return false;
  destination = std::move(next);
  return true;
}

bool PlanRefinement(const Mesh &mesh,
                    const Topology &tree,
                    const std::vector<double> &velocity,
                    double width,
                    std::array<double, 3> origin,
                    const RefinementOptions &options,
                    Topology &destination,
                    RefinementReport &report) {
  Requests requests{tree.Roots(), width, origin, options, {}, 0};
  size_t n = mesh.CellsPerBlock(), per_block = n * n * n;
  if (!requests.Validate() || !requests.Protect() || !per_block ||
      mesh.Cells().size() != tree.Leaves().size() * per_block || velocity.size() != mesh.Faces().size())
    return false;
  for (double u : velocity)
    if (!std::isfinite(u))
      return false;
  std::vector<std::array<double, 3>> centre(mesh.Cells().size());
  std::vector<bool> wall(tree.Leaves().size());
  for (size_t i = 0; i < centre.size(); ++i) {
    std::array<double, 3> area{};
    for (int k = mesh.Offsets()[i]; k < mesh.Offsets()[i + 1]; ++k) {
      int id = std::abs(mesh.Incidence()[k]) - 1;
      auto f = mesh.Faces()[id];
      double weight = f.area * f.aperture;
      area[f.axis] += weight;
      centre[i][f.axis] += weight * velocity[id];
      if (f.aperture < 1)
        wall[i / per_block] = true;
    }
    for (int a = 0; a < 3; ++a)
      if (area[a] > 0)
        centre[i][a] /= area[a];
  }
  std::vector<double> indicator(tree.Leaves().size());
  for (size_t i = 0; i < centre.size(); ++i) {
    if (!mesh.Cells()[i].active)
      continue;
    double variation = 0, weight = 0;
    for (int k = mesh.Offsets()[i]; k < mesh.Offsets()[i + 1]; ++k) {
      auto f = mesh.Faces()[std::abs(mesh.Incidence()[k]) - 1];
      int j = f.low == int(i) ? f.high : f.low;
      if (j < 0 || f.aperture == 0)
        continue;
      double d2 = 0, v2 = 0;
      for (int a = 0; a < 3; ++a) {
        double dx = mesh.Cells()[i].centre[a] - mesh.Cells()[j].centre[a], du = centre[i][a] - centre[j][a];
        d2 += dx * dx;
        v2 += du * du;
      }
      double area = f.area * f.aperture;
      variation += area * v2 / d2;
      weight += area;
    }
    double value = weight > 0 ? mesh.Cells()[i].width * std::sqrt(variation / weight) / options.reference_speed : 0;
    if (!std::isfinite(value))
      return false;
    indicator[i / per_block] = std::max(indicator[i / per_block], value);
  }
  RefinementReport next_report;
  next_report.old_blocks = tree.Leaves().size();
  for (size_t i = 0; i < tree.Leaves().size(); ++i) {
    auto b = tree.Leaves()[i];
    if (b.level > options.max_level)
      return false;
    next_report.max_indicator = std::max(next_report.max_indicator, indicator[i]);
    int level = std::max(0, b.level - 1);
    double padding = 0;
    if (indicator[i] >= options.coarsen_threshold || wall[i])
      level = b.level;
    if (indicator[i] > options.refine_threshold && b.level < options.max_level) {
      level = b.level + 1;
      ++next_report.marked_refine;
    }
    if (level >= b.level && indicator[i] >= options.coarsen_threshold)
      padding = options.buffer_blocks * width / (1 << b.level);
    if (!requests.Box(Bounds(b, width, origin), level, padding))
      return false;
  }
  Topology next;
  if (!requests.Build(next))
    return false;
  next_report.new_blocks = next.Leaves().size();
  next_report.changed = next.Leaves() != tree.Leaves() || next.MaxLevel() != tree.MaxLevel();
  destination = std::move(next);
  report = next_report;
  return true;
}
}  // namespace wind3d::amr
