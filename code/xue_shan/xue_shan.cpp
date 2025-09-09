#include "xue_shan/xue_shan.h"

namespace XS {
void PyBind(pybind11::module_ &m) {
  pybind11::module_ visualizer = m.def_submodule("visualizer", "Visualizer module for xue_shan");
  visualizer::PyBind(visualizer);
  pybind11::module_ solver = m.def_submodule("solver", "Solver module for xue_shan");
  solver::PyBind(solver);
}
}  // namespace XS
