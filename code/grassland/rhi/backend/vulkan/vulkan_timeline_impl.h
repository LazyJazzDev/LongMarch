#pragma once
#include "grassland/rhi/backend/backend_timeline_state.h"
#include "grassland/rhi/backend/vulkan/vulkan_timeline.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
class VulkanTimelineImpl final : public Object<IVulkanTimeline> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(ITimeline, IVulkanTimeline, IBackendTimeline, ITimeline)
  static ObjectPtr<IVulkanTimeline> Create(IVulkanDevice *device);
  ~VulkanTimelineImpl() override;
  bool GetCompletedValue(uint64_t &value) const override;
  bool WaitHost(uint64_t value) const override;
  VkSemaphore NativeSemaphore() const;

 private:
  explicit VulkanTimelineImpl(IVulkanDevice *device);
  VkDevice device_{};
  VkSemaphore semaphore_{};

 private:
  TimelineState state_Timeline_;

  TimelineState &StateTimeline() override {
    return state_Timeline_;
  }

  const TimelineState &StateTimeline() const override {
    return state_Timeline_;
  }
};
}  // namespace grassland::rhi::backend::vulkan
