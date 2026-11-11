#pragma once
#include <list>

#include "grassland/rhi/backend/backend_command_context.h"
#include "grassland/rhi/backend/backend_device.h"
#include "grassland/rhi/backend/backend_timeline.h"
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_queue.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::IBackendQueue)

namespace grassland::rhi::backend {

struct QueueState;

class IBackendQueue : public rhi::IQueue {
 protected:
  void InitializeQueue(IBackendDevice *device,
                       QueueType type,
                       std::shared_ptr<std::mutex> mutex = std::make_shared<std::mutex>());
  void DestroyQueue();

 public:
  IBackendDevice *Owner() const;
  virtual IBackendTimeline *GetTimeline() const = 0;
  QueueType GetType() const override = 0;
  QueueType CommonGetType() const;
  std::mutex &QueueMutex();
  // Called under QueueMutex, in native submission order.
  uint64_t NextSerial();
  ObjectPtr<IBackendSubmissionToken> Submit(const ObjectPtr<IBackendCommandContext> &commands, const SubmitInfo &info);
  // Called only by the device collector; returns whether this queue is drained.
  bool CollectCompleted(uint64_t completed, bool timeline_valid);

 protected:
  virtual QueueState &StateQueue() = 0;
  virtual const QueueState &StateQueue() const = 0;
};

}  // namespace grassland::rhi::backend
