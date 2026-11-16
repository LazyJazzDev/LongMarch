#include "amr_partition.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <set>

namespace wind3d::amr {
bool PressurePartition::Build(const Mesh &mesh, const Topology &tree, const FaceMass &mass, int ranks) {
  const auto &leaves = tree.Leaves();
  size_t n = mesh.CellsPerBlock(), nc = mesh.Cells().size(), nf = mesh.Faces().size();
  if (ranks < 1 || size_t(ranks) > leaves.size() || n < 2 || n > 64 ||
      leaves.size() > std::numeric_limits<size_t>::max() / (n * n * n) || nc != leaves.size() * n * n * n ||
      nf > size_t(std::numeric_limits<int>::max()) || nc > size_t(std::numeric_limits<int>::max()) ||
      mass.Offsets().size() != nf + 1 || mesh.Offsets().size() != nc + 1 ||
      mass.Columns().size() != mass.Weights().size())
    return false;
  std::vector<size_t> order(leaves.size());
  std::iota(order.begin(), order.end(), 0);
  std::vector<std::array<uint64_t, 3>> centres(leaves.size());
  for (size_t i = 0; i < leaves.size(); ++i) {
    auto b = leaves[i];
    uint64_t scale = uint64_t(1) << (tree.MaxLevel() - b.level);
    centres[i] = {(2 * uint64_t(b.x) + 1) * scale, (2 * uint64_t(b.y) + 1) * scale, (2 * uint64_t(b.z) + 1) * scale};
  }
  // Compare interleaved bits directly: valid topology coordinates may require
  // 96 bits, so a 64-bit Morton key would silently truncate large domains.
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    for (int bit = 31; bit >= 0; --bit)
      for (int axis = 2; axis >= 0; --axis) {
        auto x = (centres[a][axis] >> bit) & 1, y = (centres[b][axis] >> bit) & 1;
        if (x != y)
          return x < y;
      }
    return a < b;
  });
  PressurePartition next;
  next.cells_.resize(ranks);
  next.faces_.resize(ranks);
  next.cell_owners_.resize(nc);
  next.face_owners_.resize(nf);
  for (size_t i = 0; i < order.size(); ++i) {
    int owner = int(uint64_t(i) * uint64_t(ranks) / order.size());
    for (size_t j = 0; j < n * n * n; ++j)
      next.cell_owners_[order[i] * n * n * n + j] = owner;
  }
  for (size_t i = 0; i < nc; ++i)
    next.cells_[next.cell_owners_[i]].push_back(int(i));
  for (size_t i = 0; i < nf; ++i) {
    auto f = mesh.Faces()[i];
    if (f.low < -1 || f.high < -1 || f.low >= int(nc) || f.high >= int(nc) || (f.low < 0 && f.high < 0))
      return false;
    int cell = f.low < 0                                                 ? f.high
               : f.high < 0                                              ? f.low
               : mesh.Cells()[f.low].width <= mesh.Cells()[f.high].width ? f.low
                                                                         : f.high;
    int owner = next.cell_owners_[cell];
    next.face_owners_[i] = owner;
    next.faces_[owner].push_back(int(i));
  }

  struct Imports {
    std::set<int> cells, faces;
  };

  std::map<std::pair<int, int>, Imports> imports;
  auto face_import = [&](int destination, int face) {
    int source = next.face_owners_[face];
    if (source != destination)
      imports[{source, destination}].faces.insert(face);
  };
  for (size_t i = 0; i < nc; ++i) {
    for (int k = mesh.Offsets()[i]; k < mesh.Offsets()[i + 1]; ++k) {
      int face = std::abs(mesh.Incidence()[k]) - 1;
      int destination = next.cell_owners_[i];
      face_import(destination, face);
      // Owned pressure rows also need neighbors across faces owned elsewhere.
      // D^T imports alone cover only the face owner's side of this stencil.
      auto f = mesh.Faces()[face];
      if (f.aperture > 0)
        for (int cell : {f.low, f.high})
          if (cell >= 0 && next.cell_owners_[cell] != destination)
            imports[{next.cell_owners_[cell], destination}].cells.insert(cell);
    }
  }
  for (size_t i = 0; i < nf; ++i) {
    int destination = next.face_owners_[i];
    auto f = mesh.Faces()[i];
    for (int cell : {f.low, f.high})
      if (cell >= 0 && next.cell_owners_[cell] != destination)
        imports[{next.cell_owners_[cell], destination}].cells.insert(cell);
    for (int k = mass.Offsets()[i]; k < mass.Offsets()[i + 1]; ++k) {
      int face = mass.Columns()[k];
      if (face < 0 || face >= int(nf))
        return false;
      face_import(destination, face);
    }
  }
  for (const auto &[pair, data] : imports)
    next.exchanges_.push_back(
        {pair.first, pair.second, {data.cells.begin(), data.cells.end()}, {data.faces.begin(), data.faces.end()}});
  *this = std::move(next);
  return true;
}
}  // namespace wind3d::amr
