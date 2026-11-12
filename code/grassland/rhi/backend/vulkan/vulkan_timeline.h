#pragma once
#include "grassland/rhi/backend/vulkan/vulkan_util.h"
#include "grassland/rhi/rhi_timeline.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::vulkan::IVulkanTimeline,
                        grassland::rhi::IObject,
                        "30a1e28a-4bf1-4d83-a693-bc60f95ec95b")

namespace grassland::rhi::backend::vulkan {
class IVulkanTimeline : public IObject {
 public:
  virtual IVulkanDevice *Owner() const = 0;
  static ObjectPtr<IVulkanTimeline> Create(IVulkanDevice *device);
  virtual VkSemaphore NativeSemaphore() const = 0;
};

}  // namespace grassland::rhi::backend::vulkan

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
