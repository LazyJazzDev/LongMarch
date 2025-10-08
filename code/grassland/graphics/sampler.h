#pragma once
#include "grassland/graphics/graphics_util.h"

namespace grassland::graphics {

class Sampler {
 public:
  virtual ~Sampler() = default;

#if defined(LONGMARCH_PYTHON_ENABLED)
  static void PybindClassRegistration(py::classh<Sampler> &c);
#endif
};

}  // namespace grassland::graphics
