#pragma once
#include "grassland/graphics/graphics_util.h"

namespace grassland::graphics {

class Sampler {
 public:
  virtual ~Sampler() = default;

  static void PybindModuleRegistration(py::module_ &m);
};

}  // namespace grassland::graphics
