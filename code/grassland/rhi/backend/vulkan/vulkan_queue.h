#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_interface_id.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::vulkan::IVulkanQueue)
#include "grassland/rhi/backend/backend_queue.h"
#include "grassland/rhi/backend/vulkan/vulkan_device.h"
#include "grassland/rhi/backend/vulkan/vulkan_timeline.h"

namespace grassland::rhi::backend::vulkan {

class IVulkanQueue : public IBackendQueue {
 public:
  static IVulkanQueue *Create(IVulkanDevice *device, QueueType type, VkQueue queue, std::shared_ptr<std::mutex> mutex);
  virtual rhi::ICommandContext *CreateCommandContext() = 0;
  virtual VkQueue NativeQueue() const = 0;
  virtual IVulkanTimeline *GetTimeline() const = 0;
  virtual VkResult Wait(uint64_t value, uint64_t timeout = UINT64_MAX) const = 0;
};

;

}  // namespace grassland::rhi::backend::vulkan
