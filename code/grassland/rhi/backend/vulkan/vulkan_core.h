#pragma once
#include "grassland/rhi/backend/backend_core.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {

class VulkanCore final : public ICore {
 public:
  static std::unique_ptr<VulkanCore> CreateVulkanCore();
  ~VulkanCore() override;
  std::vector<PhysicalDevice> GetPhysicalDevices() const override;
  std::unique_ptr<IDevice> CreateDevice(int32_t device_idx, const ObjectPtr<ICore> &owner) const override;

 private:
  struct Adapter {
    VkPhysicalDevice native;
    uint32_t api_version;
    PhysicalDevice info;
  };

  VkInstance instance_{VK_NULL_HANDLE};
  std::vector<Adapter> adapters_;
};

}  // namespace grassland::rhi::backend::vulkan
