#pragma once
#include "grassland/rhi/backend/backend_timeline.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
class VulkanDevice;

class VulkanTimeline final : public Timeline {
 public:
  LONGMARCH_RHI_QUERY(VulkanTimeline, Timeline)

  static ObjectPtr<VulkanTimeline> Create(VulkanDevice *device);
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
