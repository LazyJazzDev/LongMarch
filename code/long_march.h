#pragma once
#include "contradium/contradium.h"
#include "grassland/grassland.h"
#include "practium/practium.h"
#include "snowberg/snowberg.h"
#include "sparkium/sparkium.h"

namespace long_march {
using namespace grassland;
using namespace snowberg;

namespace sparkium = sparkium;
namespace contradium = contradium;
namespace practium = practium;

#if defined(LONGMARCH_PYTHON_ENABLED)
void PybindModuleRegistration(py::module_ &m);
#endif

}  // namespace long_march
