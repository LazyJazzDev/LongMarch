#pragma once
#include "grassland/graphics/graphics_util.h"

namespace grassland::graphics {

class AccelerationStructure {
 public:
  virtual ~AccelerationStructure() = default;
  virtual int UpdateInstances(const std::vector<RayTracingInstance> &instances) = 0;
  virtual int UpdateInstances(const std::vector<std::pair<AccelerationStructure *, glm::mat4x3>> &instances);

  RayTracingInstance MakeInstance(const glm::mat4x3 &transform,
                                  uint32_t instance_id = 0,
                                  uint32_t instance_mask = 0xFF,
                                  uint32_t instance_hit_group_offset = 0,
                                  RayTracingInstanceFlag instance_flags = RAYTRACING_INSTANCE_FLAG_NONE);
};

}  // namespace grassland::graphics
