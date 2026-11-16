#include "amr_pressure_hierarchy.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>

namespace wind3d::amr {
bool PressureHierarchy::Build(const Mesh &mesh, const std::vector<int> &fine_owners, int ranks, int coarse_limit) {
  const auto &cells = mesh.Cells();
  if (cells.empty() || cells.size() != fine_owners.size() || ranks < 1 || coarse_limit < 1 ||
      cells.size() > size_t(std::numeric_limits<int>::max()) ||
      std::any_of(fine_owners.begin(), fine_owners.end(), [ranks](int r) { return r < 0 || r >= ranks; }))
    return false;
  using Rows = std::vector<std::map<int, double>>;
  Rows rows(cells.size());
  for (const auto &f : mesh.Faces()) {
    if (f.aperture <= 0 || !((f.low >= 0 && f.high >= 0) || (f.axis == 0 && f.high < 0)))
      continue;
    double w = f.area * f.aperture / f.distance;
    for (auto [i, j] : {std::pair{f.low, f.high}, std::pair{f.high, f.low}})
      if (i >= 0) {
        rows[i][i] += w;
        if (j >= 0)
          rows[i][j] -= w;
      }
  }
  std::array<double, 3> origin{INFINITY, INFINITY, INFINITY};
  double width = INFINITY;
  std::vector<std::array<double, 3>> centres;
  std::vector<double> volumes;
  for (size_t i = 0; i < cells.size(); ++i) {
    auto c = cells[i];
    centres.push_back(c.centre);
    volumes.push_back(c.active ? c.width * c.width * c.width : 0);
    width = std::min(width, c.width);
    for (int a = 0; a < 3; ++a)
      origin[a] = std::min(origin[a], c.centre[a] - .5 * c.width);
    if (!c.active)
      rows[i] = {{int(i), 1}};
  }
  std::vector<PressureLevel> levels;
  auto owners = fine_owners;
  for (int depth = 0; depth < 32; ++depth) {
    PressureLevel level;
    level.owners = owners;
    level.owned_cells.resize(ranks);
    level.offsets.push_back(0);
    level.identity = true;
    for (size_t i = 0; i < rows.size(); ++i) {
      auto it = rows[i].find(int(i));
      if (it == rows[i].end() || !(it->second > 0) || !std::isfinite(it->second))
        return false;
      level.diagonal.push_back(it->second);
      level.owned_cells[owners[i]].push_back(int(i));
      for (auto [j, w] : rows[i]) {
        if (!std::isfinite(w) || j < 0 || j >= int(rows.size()))
          return false;
        if (w == 0)
          continue;
        if (level.columns.size() >= size_t(std::numeric_limits<int>::max()))
          return false;
        level.columns.push_back(j);
        level.weights.push_back(w);
        if (j != int(i) || w != 1)
          level.identity = false;
      }
      level.offsets.push_back(int(level.columns.size()));
    }
    if (rows.size() <= size_t(coarse_limit) || level.identity) {
      levels.push_back(std::move(level));
      break;
    }
    width *= 2;
    std::map<std::array<long long, 3>, int> groups;
    std::vector<std::array<double, 3>> next_centres;
    std::vector<double> next_volumes;
    level.parent.assign(rows.size(), -1);
    for (size_t i = 0; i < rows.size(); ++i)
      if (volumes[i] > 0) {
        std::array<long long, 3> key;
        for (int a = 0; a < 3; ++a)
          key[a] = static_cast<long long>(std::floor((centres[i][a] - origin[a]) / width + 1e-9));
        auto [it, inserted] = groups.emplace(key, int(groups.size()));
        int j = it->second;
        level.parent[i] = j;
        if (inserted) {
          next_centres.push_back({});
          next_volumes.push_back(0);
        }
        next_volumes[j] += volumes[i];
        for (int a = 0; a < 3; ++a)
          next_centres[j][a] += volumes[i] * centres[i][a];
      }
    if (groups.empty())
      return false;
    Rows coarse(groups.size());
    for (size_t i = 0; i < rows.size(); ++i)
      if (level.parent[i] >= 0)
        for (auto [j, w] : rows[i])
          if (level.parent[j] >= 0)
            coarse[level.parent[i]][level.parent[j]] += w;
    owners.resize(groups.size());
    for (size_t i = 0; i < groups.size(); ++i) {
      owners[i] = int(i * size_t(ranks) / groups.size());
      for (int a = 0; a < 3; ++a)
        next_centres[i][a] /= next_volumes[i];
    }
    levels.push_back(std::move(level));
    rows = std::move(coarse);
    centres = std::move(next_centres);
    volumes = std::move(next_volumes);
  }
  if (levels.empty() || !levels.back().parent.empty())
    return false;
  using Imports = std::map<std::pair<int, int>, std::set<int>>;
  std::vector<Imports> imports(levels.size());
  auto need = [&](size_t depth, int destination, int cell) {
    int source = levels[depth].owners[cell];
    if (source != destination)
      imports[depth][{source, destination}].insert(cell);
  };
  for (size_t depth = 0; depth < levels.size(); ++depth) {
    const auto &level = levels[depth];
    for (size_t i = 0; i < level.owners.size(); ++i) {
      for (int k = level.offsets[i]; k < level.offsets[i + 1]; ++k)
        need(depth, level.owners[i], level.columns[k]);
      if (!level.parent.empty() && level.parent[i] >= 0) {
        int parent = level.parent[i];
        need(depth, levels[depth + 1].owners[parent], int(i));
        need(depth + 1, level.owners[i], parent);
      }
    }
  }
  for (size_t depth = 0; depth < levels.size(); ++depth)
    for (const auto &[pair, ids] : imports[depth])
      levels[depth].exchanges.push_back({pair.first, pair.second, {ids.begin(), ids.end()}});
  levels_ = std::move(levels);
  return true;
}
}  // namespace wind3d::amr
