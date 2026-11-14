#include "amr_refinement.h"

#include <cmath>
#include <iostream>

int main() {
  using namespace wind3d::amr;
  RefinementOptions options;
  options.max_level = 2;
  options.capacity = 10000;
  options.buffer_blocks = 0;
  options.protected_regions.push_back({{.2, .2, .2}, {.4, .4, .4}});
  Topology protected_tree;
  if (!ProtectedTopology({3, 2, 2}, 1, {0, 0, 0}, options, protected_tree))
    return 1;
  int protected_leaf = protected_tree.Find(1, 1, 1);
  if (protected_leaf < 0 || protected_tree.Leaves()[protected_leaf].level != 2)
    return 2;
  Mesh mesh;
  if (!mesh.Build(protected_tree, 4, 1))
    return 3;
  std::vector<double> u(mesh.Faces().size(), .4);
  Topology next;
  RefinementReport report;
  if (!PlanRefinement(mesh, protected_tree, u, 1, {0, 0, 0}, options, next, report) ||
      next.Leaves() != protected_tree.Leaves() || report.max_indicator > 1e-14)
    return 4;
  options.protected_regions.clear();
  options.max_level = 1;
  Topology coarse;
  if (!coarse.Regrid({3, 2, 2}, 1, 10000, [](Block) { return false; }) || !mesh.Build(coarse, 4, 1))
    return 5;
  u.resize(mesh.Faces().size());
  for (size_t i = 0; i < u.size(); ++i)
    u[i] = mesh.Faces()[i].axis == 0 ? 3 * mesh.Faces()[i].centre[1] : 0;
  if (!PlanRefinement(mesh, coarse, u, 1, {0, 0, 0}, options, next, report) || !report.changed || !report.marked_refine)
    return 6;
  auto fine = next;
  if (!mesh.Build(fine, 4, 1))
    return 7;
  u.resize(mesh.Faces().size());
  // Select thresholds around the measured indicator to exercise hysteresis.
  for (size_t i = 0; i < u.size(); ++i)
    u[i] = mesh.Faces()[i].axis == 0 ? mesh.Faces()[i].centre[1] : 0;
  options.refine_threshold = 1;
  options.coarsen_threshold = .001;
  if (!PlanRefinement(mesh, fine, u, 1, {0, 0, 0}, options, next, report) || next.Leaves() != fine.Leaves())
    return 8;
  u.assign(u.size(), .4);
  if (!PlanRefinement(mesh, fine, u, 1, {0, 0, 0}, options, next, report) || next.Leaves().size() != 12)
    return 9;
  auto saved = next.Leaves();
  auto saved_report = report;
  options.capacity = 12;
  options.protected_regions.push_back({{.2, .2, .2}, {.4, .4, .4}});
  if (PlanRefinement(mesh, fine, u, 1, {0, 0, 0}, options, next, report) || next.Leaves() != saved ||
      report.new_blocks != saved_report.new_blocks)
    return 10;
  options.capacity = 10000;
  options.protected_regions.clear();
  if (!mesh.Embed([](std::array<double, 3> p) {
        double x = p[0] - .5, y = p[1] - .5, z = p[2] - .5;
        return std::sqrt(x * x + y * y + z * z) - .2;
      }))
    return 11;
  u.assign(mesh.Faces().size(), 0);
  if (!PlanRefinement(mesh, fine, u, 1, {0, 0, 0}, options, next, report))
    return 12;
  int wall_leaf = next.Find(1, 1, 1);
  if (wall_leaf < 0 || next.Leaves()[wall_leaf].level != 1)
    return 13;
  std::cout
      << "PASS protected finest regions, flow refinement, hysteresis, calm-flow coarsening, wall-spacing retention and capacity rollback\n";
}
