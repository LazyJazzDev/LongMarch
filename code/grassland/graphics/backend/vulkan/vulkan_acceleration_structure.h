#pragma once
#include "grassland/graphics/backend/vulkan/vulkan_util.h"

namespace grassland::graphics::backend {
class VulkanAccelerationStructure : public AccelerationStructure {
 public:
  VulkanAccelerationStructure(VulkanCore *core,
                              VkAccelerationStructureKHR as,
                              VkBuffer buffer,
                              VmaAllocation allocation,
                              VkDeviceSize buffer_size,
                              VkDeviceAddress device_address,
                              int num_instance);
  ~VulkanAccelerationStructure() override;
  int UpdateInstances(const std::vector<RayTracingInstance> &instances) override;

  VkAccelerationStructureKHR Handle() const {
    return as_;
  }

  VkDeviceAddress DeviceAddress() const {
    return device_address_;
  }

 private:
  VulkanCore *core_;
  VkAccelerationStructureKHR as_{VK_NULL_HANDLE};
  VkBuffer buffer_{VK_NULL_HANDLE};
  VmaAllocation allocation_{VK_NULL_HANDLE};
  VkDeviceSize buffer_size_{};
  VkDeviceAddress device_address_{};
  int num_instance_{};
};
}  // namespace grassland::graphics::backend
