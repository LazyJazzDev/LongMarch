#include "grassland/grassland.h"

namespace grassland {

void PyBind(pybind11::module_ &m) {
  m.doc() = "Grassland library.";
  pybind11::module_ m_graphics = m.def_submodule("graphics", "Graphics module.");
  graphics::PyBind(m_graphics);
}

}  // namespace grassland
