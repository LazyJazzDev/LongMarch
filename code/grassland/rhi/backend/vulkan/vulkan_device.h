#pragma once
#include "grassland/rhi/backend/backend_device.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {

class VulkanDevice final : public IDevice {
 public:
  static std::unique_ptr<VulkanDevice> CreateVulkanDevice(PhysicalDevice info,
                                                          const ObjectRef &core,
                                                          VkPhysicalDevice physical_device,
                                                          uint32_t api_version);
  ~VulkanDevice() override;

 private:
  VulkanDevice(PhysicalDevice info, const ObjectRef &core);
  VkDevice device_{VK_NULL_HANDLE};
};

}  // namespace grassland::rhi::backend::vulkan
