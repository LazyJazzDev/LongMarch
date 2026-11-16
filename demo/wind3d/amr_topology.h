#pragma once

#include <array>
#include <functional>
#include <vector>

namespace wind3d::amr {
// All blocks contain the same number of MAC cells; coordinates are block indices
// at this level. Adjacent leaves differ by at most one level.
struct Block {
  int level{}, x{}, y{}, z{};

  bool operator==(const Block &other) const {
    return level == other.level && x == other.x && y == other.y && z == other.z;
  }
};

class Topology {
 public:
  using Refine = std::function<bool(const Block &)>;
  // Rebuild into a temporary tree, balance, then publish. False leaves the old
  // topology intact (including on capacity exhaustion).
  bool Regrid(std::array<int, 3> roots, int max_level, size_t capacity, const Refine &refine);

  // Restore an exact leaf cover (input order is arbitrary). Reject duplicates,
  // holes, overlaps, invalid coordinates and unbalanced covers; never silently
  // refine a saved topology. Failure preserves the current topology.
  bool Restore(std::array<int, 3> roots, int max_level, size_t capacity, const std::vector<Block> &leaves);

  const std::vector<Block> &Leaves() const {
    return leaves_;
  }

  std::array<int, 3> Roots() const {
    return roots_;
  }

  int MaxLevel() const {
    return max_level_;
  }

  // Neighbor through the face centre, in finest-level block coordinates.
  // Returns -1 outside the domain. Fine-side queries cover coarse/fine faces.
  int Neighbor(size_t leaf, int axis, int direction) const;

  // Point lookup in finest-level block coordinates, -1 outside the domain.
  int Find(int x, int y, int z) const;

 private:
  std::array<int, 3> roots_{};
  int max_level_{};
  std::vector<Block> leaves_;
};
}  // namespace wind3d::amr
