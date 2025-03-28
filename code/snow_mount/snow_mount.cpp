#include "snow_mount/snow_mount.h"

namespace snow_mount {
void PyBind(pybind11::module_ &m) {
  pybind11::module_ visualizer = m.def_submodule("visualizer", "Visualizer module for snow_mount");
  visualizer::PyBind(visualizer);
  pybind11::module_ solver = m.def_submodule("solver", "Solver module for snow_mount");
  solver::PyBind(solver);
}
}  // namespace snow_mount
