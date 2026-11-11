#pragma once
#include "grassland/rhi/backend/metal/metal_queue.h"
#include "grassland/rhi/backend/metal/metal_util.h"

namespace grassland::rhi::backend::metal {
class MetalQueueImpl final : public Object<IMetalQueue> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(rhi::IQueue, IMetalQueue, IBackendQueue, rhi::IQueue)
  rhi::ICommandContext *CreateCommandContext() override;
  static IMetalQueue *Create(IMetalDevice *device, QueueType type, MTL::CommandBufferDescriptor *descriptor);
  MTL::CommandQueue *NativeQueue() const;
  NS::SharedPtr<MTL::CommandBuffer> CreateCommandBuffer();
  bool WaitIdle();
  IMetalTimeline *GetTimeline() const override;

 private:
  MetalQueueImpl(IMetalDevice *device, QueueType type);
  NS::SharedPtr<MTL::CommandQueue> queue_;
  ObjectPtr<IMetalTimeline> timeline_;
  NS::SharedPtr<MTL::CommandBufferDescriptor> descriptor_;

 public:
  ~MetalQueueImpl() override = default;

  ObjectPtr<IBackendSubmissionToken> Submit(const ObjectPtr<IBackendCommandContext> &commands,
                                            const SubmitInfo &info) override;

  // Called only by the device collector; returns whether this queue is drained.
  bool CollectCompleted(uint64_t completed, bool timeline_valid) override;

 private:
  void InitializeQueue(IBackendDevice *device,
                       QueueType type,
                       std::shared_ptr<std::mutex> mutex = std::make_shared<std::mutex>());

 public:
  std::mutex &QueueMutex() override;

  // Called under QueueMutex, in native submission order.
  uint64_t NextSerial() override;

  IBackendDevice *Owner() const override;

  QueueType GetType() const override;

 private:
  struct Submission {
    ObjectPtr<IBackendCommandContext> commands;
    ObjectPtr<IBackendSubmissionToken> completion;
  };

  std::list<Submission> pending_{};
  IBackendDevice *device_{};
  QueueType type_{};
  // Logical queues that share a native queue also share its host synchronization.
  std::shared_ptr<std::mutex> mutex_{};
  uint64_t serial_{};
};
}  // namespace grassland::rhi::backend::metal
