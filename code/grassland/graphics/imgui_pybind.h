#pragma once
#include "grassland/graphics/graphics_util.h"

namespace grassland::graphics {
#if defined(LONGMARCH_PYTHON_ENABLED)
void PybindImGuiRegistration(py::module_ &m);
#endif
}  // namespace grassland::graphics
