#include "amr_mesh.h"

#include <cmath>
#include <limits>

namespace wind3d::amr {
bool Mesh::Build(const Topology &tree, int n, double root_width, std::array<double, 3> origin) {
  if (n < 2 || n > 64 || n % 2 || !std::isfinite(root_width) || root_width <= 0 || tree.Leaves().empty())
    return false;
  for (double x : origin)
    if (!std::isfinite(x))
      return false;
  const size_t per_block = size_t(n) * n * n;
  // A cell has at most 24 face patches, and incidence uses signed int IDs.
  if (tree.Leaves().size() > size_t(std::numeric_limits<int>::max() / 24) / per_block)
    return false;
  for (int r : tree.Roots())
    if (r > (std::numeric_limits<int>::max() >> tree.MaxLevel()) / n)
      return false;
  const double unit = root_width / (n * (1 << tree.MaxLevel()));
  std::vector<MeshCell> cells(tree.Leaves().size() * per_block);
  std::vector<MeshFace> faces;
  auto locate = [&](std::array<int, 3> p) {
    for (int a = 0; a < 3; ++a)
      if (p[a] < 0 || p[a] >= tree.Roots()[a] * n * (1 << tree.MaxLevel()))
        return -1;
    int block = tree.Find(p[0] / n, p[1] / n, p[2] / n);
    if (block < 0)
      return -1;
    const Block b = tree.Leaves()[block];
    int scale = 1 << (tree.MaxLevel() - b.level);
    int x = p[0] / scale - b.x * n, y = p[1] / scale - b.y * n, z = p[2] / scale - b.z * n;
    return int(block * per_block) + (z * n + y) * n + x;
  };
  for (size_t block = 0; block < tree.Leaves().size(); ++block) {
    const Block b = tree.Leaves()[block];
    const int scale = 1 << (tree.MaxLevel() - b.level);
    for (int z = 0; z < n; ++z)
      for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x) {
          int id = int(block * per_block) + (z * n + y) * n + x;
          std::array<int, 3> lower{(b.x * n + x) * scale, (b.y * n + y) * scale, (b.z * n + z) * scale};
          cells[id].width = scale * unit;
          for (int a = 0; a < 3; ++a)
            cells[id].centre[a] = origin[a] + (lower[a] + scale * .5) * unit;
          for (int axis = 0; axis < 3; ++axis) {
            auto probe = lower;
            for (int a = 0; a < 3; ++a)
              probe[a] += scale / 2;
            probe[axis] = lower[axis] + scale;
            int neighbor = locate(probe);
            const bool finer = neighbor >= 0 && tree.Leaves()[neighbor / int(per_block)].level > b.level;
            const int divisions = finer ? 2 : 1;
            const int patch_scale = scale / divisions;
            const int u = (axis + 1) % 3, v = (axis + 2) % 3;
            for (int j = 0; j < divisions; ++j)
              for (int i = 0; i < divisions; ++i) {
                auto p = probe;
                p[u] = lower[u] + i * patch_scale + patch_scale / 2;
                p[v] = lower[v] + j * patch_scale + patch_scale / 2;
                int other = locate(p);
                MeshFace face;
                face.low = id;
                face.high = other;
                face.axis = axis;
                face.centre[axis] = origin[axis] + probe[axis] * unit;
                face.centre[u] = origin[u] + (lower[u] + (i + .5) * patch_scale) * unit;
                face.centre[v] = origin[v] + (lower[v] + (j + .5) * patch_scale) * unit;
                face.area = patch_scale * unit * patch_scale * unit;
                int other_scale = other < 0 ? 0 : 1 << (tree.MaxLevel() - tree.Leaves()[other / int(per_block)].level);
                face.distance = .5 * (scale + other_scale) * unit;
                faces.push_back(face);
              }
            if (lower[axis] == 0) {
              MeshFace face;
              face.low = -1;
              face.high = id;
              face.axis = axis;
              face.centre = cells[id].centre;
              face.centre[axis] = origin[axis];
              face.area = scale * unit * scale * unit;
              face.distance = .5 * scale * unit;
              faces.push_back(face);
            }
          }
        }
  }
  std::vector<int> offsets(cells.size() + 1);
  for (const auto &f : faces) {
    if (f.low >= 0)
      ++offsets[f.low + 1];
    if (f.high >= 0)
      ++offsets[f.high + 1];
  }
  for (size_t i = 1; i < offsets.size(); ++i)
    offsets[i] += offsets[i - 1];
  auto cursor = offsets;
  std::vector<int> incidence(offsets.back());
  for (size_t i = 0; i < faces.size(); ++i) {
    if (faces[i].low >= 0)
      incidence[cursor[faces[i].low]++] = int(i) + 1;
    if (faces[i].high >= 0)
      incidence[cursor[faces[i].high]++] = -int(i) - 1;
  }
  embedded_ = false;
  cells_per_block_ = n;
  cells_ = std::move(cells);
  faces_ = std::move(faces);
  offsets_ = std::move(offsets);
  incidence_ = std::move(incidence);
  return true;
}
}  // namespace wind3d::amr
