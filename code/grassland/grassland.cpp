#include "grassland/grassland.h"

namespace grassland {

void PyBind(pybind11::module_ &m) {
  m.doc() = "Grassland library.";
  pybind11::module_ m_graphics = m.def_submodule("graphics", "Graphics module.");
  graphics::PyBind(m_graphics);
  pybind11::module_ m_math = m.def_submodule("math", "Math module.");
  PyBindMath(m_math);
}

}  // namespace grassland
