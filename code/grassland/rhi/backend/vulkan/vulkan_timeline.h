#pragma once
#include "grassland/rhi/backend/backend_timeline.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::vulkan::IVulkanTimeline)

namespace grassland::rhi::backend::vulkan {
class IVulkanTimeline : public IBackendTimeline {
 public:
  static ObjectPtr<IVulkanTimeline> Create(IVulkanDevice *device);
  virtual bool GetCompletedValue(uint64_t &value) const = 0;
  virtual bool WaitHost(uint64_t value) const = 0;
  virtual VkSemaphore NativeSemaphore() const = 0;
};

;

}  // namespace grassland::rhi::backend::vulkan
