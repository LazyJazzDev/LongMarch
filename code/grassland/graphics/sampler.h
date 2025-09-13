#pragma once
#include "grassland/graphics/graphics_util.h"

namespace grassland::graphics {

class Sampler {
 public:
  virtual ~Sampler() = default;

  static void PybindClassRegistration(py::classh<Sampler> &c);
};

}  // namespace grassland::graphics
