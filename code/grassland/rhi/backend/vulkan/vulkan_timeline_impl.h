#pragma once
#include "grassland/rhi/backend/vulkan/vulkan_timeline.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
class VulkanTimelineImpl final : public Object<rhi::ITimeline, IVulkanTimeline> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(ITimeline, IVulkanTimeline, IBackendTimeline, ITimeline)

  static ObjectPtr<IVulkanTimeline> Create(IVulkanDevice *device);

  ~VulkanTimelineImpl() override;

  bool GetCompletedValue(uint64_t &value) const override;

  bool WaitHost(uint64_t value) const override;

  VkSemaphore NativeSemaphore() const;

  IBackendDevice *Owner() const override {
    return owner_;
  }

 private:
  explicit VulkanTimelineImpl(IVulkanDevice *device);

  VkDevice device_{};

  VkSemaphore semaphore_{};

  void InitializeTimeline(IBackendDevice *device) {
    owner_ = device;
  }

  IBackendDevice *owner_{};
};
}  // namespace grassland::rhi::backend::vulkan
