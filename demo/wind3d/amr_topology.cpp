#include "amr_topology.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <unordered_set>

namespace wind3d::amr {
namespace {
struct Hash {
  size_t operator()(const Block &b) const {
    size_t h = size_t(b.level);
    for (int coordinate : {b.x, b.y, b.z})
      h = (h * 1000003) ^ size_t(coordinate);
    return h;
  }
};

using LeafSet = std::unordered_set<Block, Hash>;

bool Split(LeafSet &leaves, const Block &b, size_t capacity) {
  if (leaves.size() > capacity || capacity - leaves.size() < 7)
    return false;
  leaves.erase(b);
  for (int z = 0; z < 2; ++z)
    for (int y = 0; y < 2; ++y)
      for (int x = 0; x < 2; ++x)
        leaves.insert({b.level + 1, 2 * b.x + x, 2 * b.y + y, 2 * b.z + z});
  return true;
}

Block Locate(const LeafSet &leaves, int max_level, const std::array<int, 3> &p) {
  for (int level = max_level; level >= 0; --level) {
    const int scale = 1 << (max_level - level);
    Block b{level, p[0] / scale, p[1] / scale, p[2] / scale};
    if (leaves.count(b))
      return b;
  }
  return {-1, 0, 0, 0};
}

std::array<int, 3> Across(Block b, int max_level, int axis, int direction) {
  const int scale = 1 << (max_level - b.level);
  std::array<int, 3> p{b.x * scale + scale / 2, b.y * scale + scale / 2, b.z * scale + scale / 2};
  const int coordinate = axis == 0 ? b.x : axis == 1 ? b.y : b.z;
  p[axis] = direction < 0 ? coordinate * scale - 1 : (coordinate + 1) * scale;
  return p;
}

bool Inside(const std::array<int, 3> &p, std::array<int, 3> roots, int max_level) {
  for (int axis = 0; axis < 3; ++axis)
    if (p[axis] < 0 || p[axis] >= roots[axis] * (1 << max_level))
      return false;
  return true;
}
}  // namespace

bool Topology::Regrid(std::array<int, 3> roots, int max_level, size_t capacity, const Refine &refine) {
  if (max_level < 0 || max_level > 12 || !refine)
    return false;
  uint64_t count = 1;
  for (int n : roots) {
    if (n <= 0 || n > (std::numeric_limits<int>::max() >> max_level))
      return false;
    if (uint64_t(n) > capacity / count)
      return false;
    count *= n;
  }
  LeafSet leaves;
  std::vector<Block> pending;
  for (int z = 0; z < roots[2]; ++z)
    for (int y = 0; y < roots[1]; ++y)
      for (int x = 0; x < roots[0]; ++x) {
        Block b{0, x, y, z};
        leaves.insert(b);
        pending.push_back(b);
      }
  while (!pending.empty()) {
    const Block b = pending.back();
    pending.pop_back();
    if (b.level == max_level || !refine(b))
      continue;
    if (!Split(leaves, b, capacity))
      return false;
    for (int z = 0; z < 2; ++z)
      for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 2; ++x)
          pending.push_back({b.level + 1, 2 * b.x + x, 2 * b.y + y, 2 * b.z + z});
  }
  // Fine-side face probes find all violations, without a quadratic pair scan.
  for (;;) {
    LeafSet split;
    for (const Block &b : leaves)
      for (int axis = 0; axis < 3; ++axis)
        for (int direction : {-1, 1}) {
          auto p = Across(b, max_level, axis, direction);
          if (!Inside(p, roots, max_level))
            continue;
          auto neighbor = Locate(leaves, max_level, p);
          if (neighbor.level < 0)
            return false;
          if (b.level > neighbor.level + 1)
            split.insert(neighbor);
        }
    if (split.empty())
      break;
    for (const Block &b : split)
      if (!Split(leaves, b, capacity))
        return false;
  }
  std::vector<Block> result(leaves.begin(), leaves.end());
  std::sort(result.begin(), result.end(), [](Block a, Block b) {
    return std::array<int, 4>{a.level, a.z, a.y, a.x} < std::array<int, 4>{b.level, b.z, b.y, b.x};
  });
  roots_ = roots;
  max_level_ = max_level;
  leaves_ = std::move(result);
  return true;
}

int Topology::Find(int x, int y, int z) const {
  if (!Inside({x, y, z}, roots_, max_level_))
    return -1;
  for (int level = max_level_; level >= 0; --level) {
    const int scale = 1 << (max_level_ - level);
    Block b{level, x / scale, y / scale, z / scale};
    auto less = [](Block a, Block b) {
      return std::array<int, 4>{a.level, a.z, a.y, a.x} < std::array<int, 4>{b.level, b.z, b.y, b.x};
    };
    auto it = std::lower_bound(leaves_.begin(), leaves_.end(), b, less);
    if (it != leaves_.end() && *it == b)
      return int(it - leaves_.begin());
  }
  return -1;
}

int Topology::Neighbor(size_t leaf, int axis, int direction) const {
  if (leaf >= leaves_.size() || axis < 0 || axis > 2 || (direction != -1 && direction != 1))
    return -1;
  auto p = Across(leaves_[leaf], max_level_, axis, direction);
  return Find(p[0], p[1], p[2]);
}
}  // namespace wind3d::amr
