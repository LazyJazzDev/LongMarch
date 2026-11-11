#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend::vulkan {
class IVulkanCore;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::vulkan::IVulkanCore)
#include "grassland/rhi/backend/backend_core.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {

class IVulkanCore : public Core {
 public:
  static IVulkanCore *CreateVulkanCore(const CoreDesc &desc);
  virtual VkInstance Instance() const = 0;
  virtual std::vector<PhysicalDevice> GetPhysicalDevices() const = 0;
  virtual rhi::IDevice *CreateDevice(int32_t device_idx) = 0;
};

;

}  // namespace grassland::rhi::backend::vulkan
