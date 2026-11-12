#pragma once
#include <list>
#include <mutex>

#include "grassland/rhi/backend/backend_submission_token.h"
#include "grassland/rhi/backend/metal/metal_command_context.h"
#include "grassland/rhi/backend/metal/metal_device.h"
#include "grassland/rhi/backend/metal/metal_timeline.h"
#include "grassland/rhi/backend/metal/metal_util.h"
#include "grassland/rhi/rhi_queue.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::metal::IMetalQueue,
                        grassland::rhi::IObject,
                        "b34deab1-e42d-43f3-b6c4-f47dc7638e30")

namespace grassland::rhi::backend::metal {

class IMetalQueue : public IObject {
 public:
  virtual std::mutex &QueueMutex() = 0;
  virtual  // Called under QueueMutex, in native submission order.
      uint64_t
      NextSerial() = 0;
  virtual ObjectPtr<IBackendSubmissionToken> Submit(const ObjectPtr<IMetalCommandContext> &commands,
                                                    const SubmitInfo &info) = 0;
  virtual  // Called only by the device collector; returns whether this queue is drained.
      bool
      CollectCompleted(uint64_t completed, bool timeline_valid) = 0;
  virtual IMetalDevice *Owner() const = 0;
  static IMetalQueue *Create(IMetalDevice *device, QueueType type, MTL::CommandBufferDescriptor *descriptor);
  virtual MTL::CommandQueue *NativeQueue() const = 0;
  virtual NS::SharedPtr<MTL::CommandBuffer> CreateCommandBuffer() = 0;
  virtual bool WaitIdle() = 0;
  virtual IMetalTimeline *GetTimeline() const = 0;
};

}  // namespace grassland::rhi::backend::metal

namespace grassland::rhi::backend::metal {
class MetalQueueImpl final : public Object<rhi::IQueue, IMetalQueue> {
 public:
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

  ObjectPtr<IBackendSubmissionToken> Submit(const ObjectPtr<IMetalCommandContext> &commands,
                                            const SubmitInfo &info) override;

  // Called only by the device collector; returns whether this queue is drained.
  bool CollectCompleted(uint64_t completed, bool timeline_valid) override;

 private:
  void InitializeQueue(IMetalDevice *device,
                       QueueType type,
                       std::shared_ptr<std::mutex> mutex = std::make_shared<std::mutex>());

 public:
  std::mutex &QueueMutex() override;

  // Called under QueueMutex, in native submission order.
  uint64_t NextSerial() override;

  IMetalDevice *Owner() const override;

  QueueType GetType() const override;

 private:
  struct Submission {
    ObjectPtr<IMetalCommandContext> commands;
    ObjectPtr<IBackendSubmissionToken> completion;
  };

  std::list<Submission> pending_{};
  IMetalDevice *device_{};
  QueueType type_{};
  // Logical queues that share a native queue also share its host synchronization.
  std::shared_ptr<std::mutex> mutex_{};
  uint64_t serial_{};
};
}  // namespace grassland::rhi::backend::metal
