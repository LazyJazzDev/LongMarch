#pragma once
#include "grassland/rhi/backend/vulkan/vulkan_core.h"

namespace grassland::rhi::backend::vulkan {
class VulkanCoreImpl final : public Object<IVulkanCore> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(rhi::ICore, IVulkanCore, Core, rhi::ICore)
  static IVulkanCore *CreateVulkanCore(const CoreDesc &desc);
  ~VulkanCoreImpl() override;
  VkInstance Instance() const;
  std::vector<PhysicalDevice> GetPhysicalDevices() const override;
  rhi::IDevice *CreateDevice(int32_t device_idx) override;

 private:
  VulkanCoreImpl();

  struct Adapter {
    VkPhysicalDevice native;
    uint32_t api_version;
    PhysicalDevice info;
  };

  VkInstance instance_{VK_NULL_HANDLE};
  VkDebugUtilsMessengerEXT debug_messenger_{};
  std::vector<Adapter> adapters_;
};
}  // namespace grassland::rhi::backend::vulkan
