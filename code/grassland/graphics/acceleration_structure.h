#pragma once
#include "grassland/graphics/graphics_util.h"

namespace grassland::graphics {

class AccelerationStructure {
 public:
  virtual ~AccelerationStructure() = default;
  virtual int UpdateInstances(const std::vector<std::pair<AccelerationStructure *, glm::mat4>> &instances) = 0;
};

}  // namespace grassland::graphics
