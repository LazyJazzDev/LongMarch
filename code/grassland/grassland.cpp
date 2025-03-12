#include "grassland/grassland.h"

namespace grassland {

void PybindModuleRegistration(pybind11::module_ &m) {
  m.doc() = "Grassland library.";
  pybind11::module_ m_graphics = m.def_submodule("graphics", "Graphics module.");
  graphics::PybindModuleRegistration(m_graphics);
}

}  // namespace grassland
