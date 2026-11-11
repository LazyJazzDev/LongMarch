#pragma once
#include "grassland/rhi/backend/backend_timeline.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::vulkan::IVulkanTimeline,
                        grassland::rhi::backend::IBackendTimeline,
                        "30a1e28a-4bf1-4d83-a693-bc60f95ec95b")

namespace grassland::rhi::backend::vulkan {
class IVulkanTimeline : public IBackendTimeline {
 public:
  static ObjectPtr<IVulkanTimeline> Create(IVulkanDevice *device);
  virtual VkSemaphore NativeSemaphore() const = 0;
};

}  // namespace grassland::rhi::backend::vulkan
