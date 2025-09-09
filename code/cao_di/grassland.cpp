#include "cao_di/grassland.h"

namespace CD {

void PyBind(pybind11::module_ &m) {
  m.doc() = "CaoDi library.";
  pybind11::module_ m_graphics = m.def_submodule("graphics", "Graphics module.");
  graphics::PyBind(m_graphics);
  pybind11::module_ m_math = m.def_submodule("math", "Math module.");
  PyBindMath(m_math);
  pybind11::module_ m_util = m.def_submodule("util", "Util module.");
  PyBindUtil(m_util);
}

}  // namespace CD
