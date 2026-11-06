#pragma once
#include "grassland/rhi/backend/backend_queue.h"
#include "grassland/rhi/backend/vulkan/vulkan_timeline.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
class VulkanDevice;

class VulkanQueue final : public Queue {
 public:
  ICommandContext *CreateCommandContext() override;
  static VulkanQueue *Create(VulkanDevice *device,
                             QueueType type,
                             VkQueue queue,
                             VulkanQueue *physical_owner = nullptr);
  ~VulkanQueue() override;
  VkQueue Native() const;
  VkSemaphore Timeline() const;
  VkResult Wait(uint64_t value, uint64_t timeout = UINT64_MAX) const;
  const void *NativeIdentity() const override;
  std::mutex &Mutex() override;

 private:
  VulkanQueue(VulkanDevice *device,
              QueueType type,
              VkQueue queue,
              VulkanQueue *physical_owner,
              ObjectPtr<VulkanTimeline> timeline);
  VkQueue queue_{};
  VulkanQueue *physical_owner_{};
};
}  // namespace grassland::rhi::backend::vulkan
