#pragma once
#include "cao_di/graphics/backend/vulkan/vulkan_util.h"

namespace CD::graphics::backend {

class VulkanAccelerationStructure : public AccelerationStructure {
 public:
  VulkanAccelerationStructure(VulkanCore *core,
                              std::unique_ptr<vulkan::AccelerationStructure> &&acceleration_structure);
  int UpdateInstances(const std::vector<RayTracingInstance> &instances) override;

  vulkan::AccelerationStructure *Handle() const {
    return acceleration_structure_.get();
  }

 private:
  VulkanCore *core_;
  std::unique_ptr<vulkan::AccelerationStructure> acceleration_structure_;
};

}  // namespace CD::graphics::backend
