#include "xue_shan/snow_mount.h"

namespace XS {
void PyBind(pybind11::module_ &m) {
  pybind11::module_ visualizer = m.def_submodule("visualizer", "Visualizer module for snow_mount");
  visualizer::PyBind(visualizer);
  pybind11::module_ solver = m.def_submodule("solver", "Solver module for snow_mount");
  solver::PyBind(solver);
}
}  // namespace XS
