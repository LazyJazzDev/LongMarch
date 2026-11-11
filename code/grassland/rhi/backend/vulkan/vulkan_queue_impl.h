#pragma once
#include "grassland/rhi/backend/backend_queue_state.h"
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/backend/vulkan/vulkan_queue.h"

namespace grassland::rhi::backend::vulkan {
class VulkanQueueImpl final : public Object<IVulkanQueue> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(rhi::IQueue, IVulkanQueue, IBackendQueue, rhi::IQueue)
  rhi::ICommandContext *CreateCommandContext() override;
  static IVulkanQueue *Create(IVulkanDevice *device, QueueType type, VkQueue queue, std::shared_ptr<std::mutex> mutex);
  VkQueue NativeQueue() const;
  IVulkanTimeline *GetTimeline() const override;
  VkResult Wait(uint64_t value, uint64_t timeout = UINT64_MAX) const;

 private:
  VulkanQueueImpl(IVulkanDevice *device, QueueType type, VkQueue queue, std::shared_ptr<std::mutex> mutex);
  VkQueue queue_{};
  ObjectPtr<IVulkanTimeline> timeline_;

 public:
  ~VulkanQueueImpl() override {
  }

 private:
  QueueState state_Queue_;

  QueueState &StateQueue() override {
    return state_Queue_;
  }

  const QueueState &StateQueue() const override {
    return state_Queue_;
  }

 public:
  QueueType GetType() const override {
    return IBackendQueue::CommonGetType();
  }
};
}  // namespace grassland::rhi::backend::vulkan
