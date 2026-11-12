#pragma once
#include "grassland/rhi/backend/vulkan/vulkan_timeline.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
class VulkanTimelineImpl final : public Object<rhi::ITimeline, IVulkanTimeline> {
 public:
  static ObjectPtr<IVulkanTimeline> Create(IVulkanDevice *device);

  ~VulkanTimelineImpl() override;

  bool GetCompletedValue(uint64_t &value) const override;

  bool WaitHost(uint64_t value) const override;

  VkSemaphore NativeSemaphore() const;

  IVulkanDevice *Owner() const override {
    return owner_;
  }

 private:
  explicit VulkanTimelineImpl(IVulkanDevice *device);

  VkDevice device_{};

  VkSemaphore semaphore_{};

  void InitializeTimeline(IVulkanDevice *device) {
    owner_ = device;
  }

  IVulkanDevice *owner_{};
};
}  // namespace grassland::rhi::backend::vulkan
