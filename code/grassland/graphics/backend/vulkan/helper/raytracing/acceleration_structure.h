#pragma once
#include "grassland/graphics/backend/vulkan/helper/buffer.h"
#include "grassland/graphics/backend/vulkan/helper/device.h"

namespace grassland::graphics::backend::vulkan {
class AccelerationStructure {
 public:
  AccelerationStructure(const class Device *device,
                        std::unique_ptr<class Buffer> buffer,
                        VkDeviceAddress device_address,
                        VkAccelerationStructureKHR as,
                        int num_instance);
  ~AccelerationStructure();
  class Buffer *Buffer() const;
  VkDeviceAddress DeviceAddress() const;

  VkAccelerationStructureKHR Handle() const {
    return as_;
  }

  VkResult UpdateInstances(const std::vector<VkAccelerationStructureInstanceKHR> &instances,
                           VkCommandPool command_pool,
                           VkQueue queue);

  VkResult UpdateInstances(const std::vector<std::pair<AccelerationStructure *, glm::mat4>> &objects,
                           VkCommandPool command_pool,
                           VkQueue queue);

 private:
  const class Device *device_{};
  std::unique_ptr<class Buffer> buffer_;
  VkDeviceAddress device_address_{};
  VkAccelerationStructureKHR as_{};
  int num_instance_;
};

VkResult BuildAccelerationStructure(const Device *device,
                                    VkAccelerationStructureGeometryKHR geometry,
                                    VkAccelerationStructureTypeKHR type,
                                    VkBuildAccelerationStructureFlagsKHR flags,
                                    VkBuildAccelerationStructureModeKHR mode,
                                    uint32_t primitive_count,
                                    VkCommandPool command_pool,
                                    VkQueue queue,
                                    VkAccelerationStructureKHR *ptr_acceleration_structure,
                                    double_ptr<Buffer> pp_buffer);

}  // namespace grassland::graphics::backend::vulkan
