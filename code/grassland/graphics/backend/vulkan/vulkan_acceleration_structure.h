#pragma once
#include "grassland/graphics/backend/vulkan/vulkan_util.h"

namespace grassland::graphics::backend {

class VulkanAccelerationStructure : public AccelerationStructure {
 public:
  VulkanAccelerationStructure(VulkanCore *core,
                              std::unique_ptr<vulkan::AccelerationStructure> &&acceleration_structure);
  int UpdateInstances(const std::vector<std::pair<AccelerationStructure *, glm::mat4>> &instances) override;

  vulkan::AccelerationStructure *Handle() const {
    return acceleration_structure_.get();
  }

 private:
  VulkanCore *core_;
  std::unique_ptr<vulkan::AccelerationStructure> acceleration_structure_;
};

}  // namespace grassland::graphics::backend
