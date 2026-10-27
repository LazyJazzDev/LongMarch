#pragma once
#include "grassland/rhi/backend/backend_core.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {

class VulkanCore final : public Core {
 public:
  BackendAPI GetBackendAPI() const override {
    return BACKEND_API_VULKAN;
  }

  static VulkanCore *CreateVulkanCore(const CoreDesc &desc);
  ~VulkanCore() override;
  VkInstance Instance() const;
  std::vector<PhysicalDevice> GetPhysicalDevices() const override;
  rhi::IDevice *CreateDevice(int32_t device_idx) override;

 private:
  VulkanCore();

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
