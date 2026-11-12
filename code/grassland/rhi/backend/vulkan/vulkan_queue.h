#pragma once
#include <list>
#include <mutex>

#include "grassland/rhi/backend/backend_submission_token.h"
#include "grassland/rhi/backend/vulkan/vulkan_command_context.h"
#include "grassland/rhi/backend/vulkan/vulkan_device.h"
#include "grassland/rhi/backend/vulkan/vulkan_timeline.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"
#include "grassland/rhi/rhi_queue.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::vulkan::IVulkanQueue,
                        grassland::rhi::IObject,
                        "09666973-0b3a-4165-bb7f-0faf38e0f01a")

namespace grassland::rhi::backend::vulkan {

class IVulkanQueue : public IObject {
 public:
  virtual std::mutex &QueueMutex() = 0;
  virtual  // Called under QueueMutex, in native submission order.
      uint64_t
      NextSerial() = 0;
  virtual ObjectPtr<IBackendSubmissionToken> Submit(const ObjectPtr<IVulkanCommandContext> &commands,
                                                    const SubmitInfo &info) = 0;
  virtual  // Called only by the device collector; returns whether this queue is drained.
      bool
      CollectCompleted(uint64_t completed, bool timeline_valid) = 0;
  virtual IVulkanDevice *Owner() const = 0;
  static IVulkanQueue *Create(IVulkanDevice *device, QueueType type, VkQueue queue, std::shared_ptr<std::mutex> mutex);
  virtual VkQueue NativeQueue() const = 0;
  virtual IVulkanTimeline *GetTimeline() const = 0;
  virtual VkResult Wait(uint64_t value, uint64_t timeout = UINT64_MAX) const = 0;
};

}  // namespace grassland::rhi::backend::vulkan

namespace grassland::rhi::backend::vulkan {
class VulkanQueueImpl final : public Object<rhi::IQueue, IVulkanQueue> {
 public:
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

  ObjectPtr<IBackendSubmissionToken> Submit(const ObjectPtr<IVulkanCommandContext> &commands,
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
    ObjectPtr<IVulkanCommandContext> commands;
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
