#pragma once
#include "grassland/graphics/interface.h"

namespace grassland::graphics {
#if defined(LONGMARCH_PYTHON_ENABLED)
void PybindModuleRegistration(py::module_ &m);
#endif
}  // namespace grassland::graphics
