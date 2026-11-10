#pragma once
#include "grassland/rhi/backend/backend_timeline.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
class VulkanDevice;

class VulkanTimeline final : public Timeline {
 public:
  static std::unique_ptr<VulkanTimeline> Create(VulkanDevice *device);
  ~VulkanTimeline() override;
  bool GetCompletedValue(uint64_t &value) const override;
  bool WaitHost(uint64_t value) const override;
  VkSemaphore NativeSemaphore() const;

 private:
  explicit VulkanTimeline(VulkanDevice *device);
  VkDevice device_{};
  VkSemaphore semaphore_{};
};

}  // namespace grassland::rhi::backend::vulkan
