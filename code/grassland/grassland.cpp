#include "grassland/grassland.h"

namespace grassland {

void PybindModuleRegistration(pybind11::module_ &m) {
  m.doc() = "Grassland library.";
}

}  // namespace grassland
