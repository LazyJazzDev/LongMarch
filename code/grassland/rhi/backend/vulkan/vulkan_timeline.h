#pragma once
#include "grassland/rhi/backend/backend_timeline.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
class VulkanDevice;

class VulkanTimeline final : public backend::Timeline {
 public:
  static VulkanTimeline *Create(VulkanDevice *device);
  ~VulkanTimeline() override;
  VkSemaphore Native() const;
  bool GetCompletedValue(uint64_t &value) const override;

  bool IsNative() const override {
    return true;
  }

  VkResult Wait(uint64_t value, uint64_t timeout = UINT64_MAX) const;

 private:
  explicit VulkanTimeline(VulkanDevice *device);
  VkSemaphore semaphore_{};
};
}  // namespace grassland::rhi::backend::vulkan
