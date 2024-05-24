#pragma once
#include "grassland/vulkan/buffer.h"
#include "grassland/vulkan/device.h"

namespace grassland::vulkan {
class AccelerationStructure {
 public:
  AccelerationStructure(const class Device *device,
                        std::unique_ptr<class Buffer> buffer,
                        VkDeviceAddress device_address,
                        VkAccelerationStructureKHR as);
  ~AccelerationStructure();
  class Buffer *Buffer() const;
  VkDeviceAddress DeviceAddress() const;
  VkAccelerationStructureKHR Handle() const {
    return as_;
  }

  VkResult UpdateInstances(
      const std::vector<std::pair<AccelerationStructure *, glm::mat4>> &objects,
      CommandPool *command_pool,
      Queue *queue);

 private:
  const class Device *device_{};
  std::unique_ptr<class Buffer> buffer_;
  VkDeviceAddress device_address_{};
  VkAccelerationStructureKHR as_{};
};
}  // namespace grassland::vulkan
