#pragma once
#include "grassland/rhi/backend/d3d12/d3d12_queue.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12QueueImpl final : public Object<rhi::IQueue, ID3D12Queue> {
 public:
  rhi::ICommandContext *CreateCommandContext() override;
  static ID3D12Queue *Create(ID3D12Device *device, QueueType type);
  ID3D12CommandQueue *NativeQueue() const;
  ID3D12Timeline *GetTimeline() const;
  bool WaitIdle();

 private:
  D3D12QueueImpl(ID3D12Device *device, QueueType type);
  ComPtr<ID3D12CommandQueue> queue_;
  ObjectPtr<ID3D12Timeline> timeline_;

 public:
  ~D3D12QueueImpl() override = default;

  ObjectPtr<IBackendSubmissionToken> Submit(const ObjectPtr<IBackendCommandContext> &commands,
                                            const SubmitInfo &info) override;

  // Called only by the device collector; returns whether this queue is drained.
  bool CollectCompleted(uint64_t completed, bool timeline_valid) override;

 private:
  void InitializeQueue(ID3D12Device *device,
                       QueueType type,
                       std::shared_ptr<std::mutex> mutex = std::make_shared<std::mutex>());

 public:
  std::mutex &QueueMutex() override;

  // Called under QueueMutex, in native submission order.
  uint64_t NextSerial() override;

  ID3D12Device *Owner() const override;

  QueueType GetType() const override;

 private:
  struct Submission {
    ObjectPtr<IBackendCommandContext> commands;
    ObjectPtr<IBackendSubmissionToken> completion;
  };

  std::list<Submission> pending_{};
  ID3D12Device *device_{};
  QueueType type_{};
  // Logical queues that share a native queue also share its host synchronization.
  std::shared_ptr<std::mutex> mutex_{};
  uint64_t serial_{};
};
}  // namespace grassland::rhi::backend::d3d12
