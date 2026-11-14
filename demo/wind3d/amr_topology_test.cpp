#include "amr_topology.h"

#include <iostream>
#include <vector>

int main() {
  using namespace wind3d::amr;
  Topology tree;
  auto local = [](Block b) { return b.x < 2 && b.y < 2 && b.z < 2; };
  if (!tree.Regrid({4, 2, 3}, 3, 10000, local))
    return 1;
  auto leaves = tree.Leaves();
  const int nx = 32, ny = 16, nz = 24;
  std::vector<int> coverage(nx * ny * nz);
  bool coarse = false, fine = false;
  for (size_t i = 0; i < leaves.size(); ++i) {
    Block b = leaves[i];
    coarse |= b.level == 0;
    fine |= b.level == 3;
    int scale = 1 << (3 - b.level);
    for (int z = b.z * scale; z < (b.z + 1) * scale; ++z)
      for (int y = b.y * scale; y < (b.y + 1) * scale; ++y)
        for (int x = b.x * scale; x < (b.x + 1) * scale; ++x)
          ++coverage[(z * ny + y) * nx + x];
    for (int axis = 0; axis < 3; ++axis)
      for (int direction : {-1, 1}) {
        int n = tree.Neighbor(i, axis, direction);
        int coordinate = axis == 0 ? b.x : axis == 1 ? b.y : b.z;
        int extent = tree.Roots()[axis] << b.level;
        bool boundary = direction < 0 ? coordinate == 0 : coordinate + 1 == extent;
        if ((n < 0) != boundary)
          return 2;
        if (n >= 0 && std::abs(b.level - leaves[n].level) > 1)
          return 3;
      }
  }
  if (!coarse || !fine)
    return 4;
  for (int count : coverage)
    if (count != 1)
      return 5;
  if (tree.Regrid({4, 2, 3}, 3, 24, local) || tree.Leaves() != leaves)
    return 6;
  // Removing refinement restores the coarse domain without holes or stale leaves.
  if (!tree.Regrid({4, 2, 3}, 3, 24, [](Block) { return false; }) || tree.Leaves().size() != 24)
    return 7;
  if (tree.Regrid({0, 2, 3}, 3, 10000, local))
    return 8;
  if (!tree.Regrid({4, 2, 3}, 3, 10000, local) || tree.Leaves() != leaves)
    return 9;
  std::cout
      << "PASS adaptive refinement/coarsening, exact coverage, 2:1 face balance, deterministic rebuild and capacity rollback; leaves="
      << leaves.size() << '\n';
}
