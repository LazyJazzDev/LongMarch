#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend {
class Queue;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::Queue)
#include <list>

#include "grassland/rhi/backend/backend_command_context.h"
#include "grassland/rhi/backend/backend_device.h"
#include "grassland/rhi/backend/backend_timeline.h"
#include "grassland/rhi/rhi_queue.h"

namespace grassland::rhi::backend {

struct QueueState;

class Queue : public rhi::IQueue {
 protected:
  void InitializeQueue(Device *device,
                       QueueType type,
                       std::shared_ptr<std::mutex> mutex = std::make_shared<std::mutex>());
  void DestroyQueue();

 public:
  Device *Owner() const;
  virtual Timeline *GetTimeline() const = 0;
  QueueType GetType() const override;
  std::mutex &QueueMutex();
  // Called under QueueMutex, in native submission order.
  uint64_t NextSerial();
  ObjectPtr<SubmissionToken> Submit(const ObjectPtr<CommandContext> &commands, const SubmitInfo &info);
  // Called only by the device collector; returns whether this queue is drained.
  bool CollectCompleted(uint64_t completed, bool timeline_valid);

 protected:
  virtual QueueState &StateQueue() = 0;
  virtual const QueueState &StateQueue() const = 0;
};

}  // namespace grassland::rhi::backend
