#pragma once
#include "grassland/grassland.h"
#include "snowberg/snowberg.h"
#include "sparks/sparks.h"

namespace long_march {
using namespace grassland;
using namespace snowberg;

namespace sparks = sparks;

void PybindModuleRegistration(py::module_ &m);

}  // namespace long_march
