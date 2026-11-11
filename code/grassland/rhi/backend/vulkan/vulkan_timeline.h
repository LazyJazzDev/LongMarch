#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend::vulkan {
class IVulkanTimeline;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::vulkan::IVulkanTimeline)
#include "grassland/rhi/backend/backend_timeline.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
class IVulkanDevice;

class IVulkanTimeline : public Timeline {
 public:
  static ObjectPtr<IVulkanTimeline> Create(IVulkanDevice *device);
  virtual bool GetCompletedValue(uint64_t &value) const = 0;
  virtual bool WaitHost(uint64_t value) const = 0;
  virtual VkSemaphore NativeSemaphore() const = 0;
};

;

}  // namespace grassland::rhi::backend::vulkan
