#pragma once
#include "grassland/rhi/backend/backend_core.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::vulkan::IVulkanCore,
                        grassland::rhi::IObject,
                        "38b5aac6-db8d-4fe0-86ac-ed06f0b3c970")

namespace grassland::rhi::backend::vulkan {

class IVulkanCore : public IObject {
 public:
  static rhi::ICore *CreateVulkanCore(const CoreDesc &desc);
  virtual VkInstance Instance() const = 0;
};

}  // namespace grassland::rhi::backend::vulkan

namespace grassland::rhi::backend::vulkan {
class VulkanCoreImpl final : public Object<rhi::ICore, IVulkanCore> {
 public:
  static rhi::ICore *CreateVulkanCore(const CoreDesc &desc);
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
    return backend::CreateDeviceByAutoSelection(this, requirement);
  }
};
}  // namespace grassland::rhi::backend::vulkan
