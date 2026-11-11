#pragma once
#include "grassland/rhi/backend/vulkan/vulkan_queue.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
class VulkanQueueImpl final : public Object<rhi::IQueue, IVulkanQueue> {
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
  ~VulkanQueueImpl() override = default;

  ObjectPtr<IBackendSubmissionToken> Submit(const ObjectPtr<IBackendCommandContext> &commands,
                                            const SubmitInfo &info) override;

  // Called only by the device collector; returns whether this queue is drained.
  bool CollectCompleted(uint64_t completed, bool timeline_valid) override;

 private:
  void InitializeQueue(IVulkanDevice *device,
                       QueueType type,
                       std::shared_ptr<std::mutex> mutex = std::make_shared<std::mutex>());

 public:
  std::mutex &QueueMutex() override;

  // Called under QueueMutex, in native submission order.
  uint64_t NextSerial() override;

  IVulkanDevice *Owner() const override;

  QueueType GetType() const override;

 private:
  struct Submission {
    ObjectPtr<IBackendCommandContext> commands;
    ObjectPtr<IBackendSubmissionToken> completion;
  };

  std::list<Submission> pending_{};
  IVulkanDevice *device_{};
  QueueType type_{};
  // Logical queues that share a native queue also share its host synchronization.
  std::shared_ptr<std::mutex> mutex_{};
  uint64_t serial_{};
};
}  // namespace grassland::rhi::backend::vulkan
