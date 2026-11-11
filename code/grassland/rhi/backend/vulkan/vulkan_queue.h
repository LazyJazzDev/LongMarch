#pragma once
#include "grassland/rhi/backend/backend_queue.h"
#include "grassland/rhi/backend/vulkan/vulkan_device.h"
#include "grassland/rhi/backend/vulkan/vulkan_timeline.h"

namespace grassland::rhi::backend::vulkan {

class VulkanQueue final : public Queue {
 public:
  LONGMARCH_RHI_QUERY(VulkanQueue, Queue)

  rhi::ICommandContext *CreateCommandContext() override;
  static VulkanQueue *Create(VulkanDevice *device, QueueType type, VkQueue queue, std::shared_ptr<std::mutex> mutex);
  VkQueue NativeQueue() const;
  VulkanTimeline *GetTimeline() const override;
  VkResult Wait(uint64_t value, uint64_t timeout = UINT64_MAX) const;

 private:
  VulkanQueue(VulkanDevice *device, QueueType type, VkQueue queue, std::shared_ptr<std::mutex> mutex);
  VkQueue queue_{};
  ObjectPtr<VulkanTimeline> timeline_;
};

}  // namespace grassland::rhi::backend::vulkan
