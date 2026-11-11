#pragma once
#include <list>

#include "grassland/rhi/backend/backend_command_context.h"
#include "grassland/rhi/backend/backend_device.h"
#include "grassland/rhi/backend/backend_timeline.h"
#include "grassland/rhi/rhi_queue.h"

namespace grassland::rhi::backend {

class Queue : public Object<rhi::IQueue> {
 public:
  LONGMARCH_RHI_QUERY(Queue, Object<rhi::IQueue>)

  Queue(Device *device, QueueType type, std::shared_ptr<std::mutex> mutex = std::make_shared<std::mutex>());
  ~Queue() override;
  Device *Owner() const;
  virtual Timeline *GetTimeline() const = 0;
  QueueType GetType() const override;
  std::mutex &QueueMutex();
  // Called under QueueMutex, in native submission order.
  uint64_t NextSerial();
  ObjectPtr<SubmissionToken> Submit(const ObjectPtr<CommandContext> &commands, const SubmitInfo &info);
  // Called only by the device collector; returns whether this queue is drained.
  bool CollectCompleted(uint64_t completed, bool timeline_valid);

 private:
  struct Submission {
    ObjectPtr<CommandContext> commands;
    ObjectPtr<SubmissionToken> completion;
  };

  std::list<Submission> pending_;
  Device *device_;
  QueueType type_;
  // Logical queues that share a native queue also share its host synchronization.
  std::shared_ptr<std::mutex> mutex_;
  uint64_t serial_{};
};

}  // namespace grassland::rhi::backend
