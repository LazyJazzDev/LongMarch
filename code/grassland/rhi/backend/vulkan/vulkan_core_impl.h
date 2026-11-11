#pragma once
#include "grassland/rhi/backend/vulkan/vulkan_core.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
class VulkanCoreImpl final : public Object<IVulkanCore> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(rhi::ICore, IVulkanCore, IBackendCore, rhi::ICore)
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

 public:
  rhi::IDevice *CreateDeviceByAutoSelection(DeviceFeatureRequirement requirement = {}) override {
    return IBackendCore::CommonCreateDeviceByAutoSelection(requirement);
  }
};
}  // namespace grassland::rhi::backend::vulkan
