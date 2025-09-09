#include "snowberg/snowberg.h"

namespace snowberg {
void PyBind(pybind11::module_ &m) {
  pybind11::module_ visualizer = m.def_submodule("visualizer", "Visualizer module for snowberg");
  visualizer::PyBind(visualizer);
  pybind11::module_ solver = m.def_submodule("solver", "Solver module for snowberg");
  solver::PyBind(solver);
}
}  // namespace snowberg
