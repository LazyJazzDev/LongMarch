#pragma once
#include <list>

#include "grassland/rhi/backend/backend_command_context.h"
#include "grassland/rhi/backend/backend_device.h"
#include "grassland/rhi/backend/backend_timeline.h"
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_queue.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::IBackendQueue)

namespace grassland::rhi::backend {

class IBackendQueue : public rhi::IQueue {
 protected:
  void DestroyQueue();

 public:
  virtual IBackendDevice *Owner() const = 0;

  virtual IBackendTimeline *GetTimeline() const = 0;

  QueueType GetType() const override = 0;

  virtual std::mutex &QueueMutex() = 0;

  virtual  // Called under QueueMutex, in native submission order.
      uint64_t
      NextSerial() = 0;

  virtual ObjectPtr<IBackendSubmissionToken> Submit(const ObjectPtr<IBackendCommandContext> &commands,
                                                    const SubmitInfo &info) = 0;

  virtual  // Called only by the device collector; returns whether this queue is drained.
      bool
      CollectCompleted(uint64_t completed, bool timeline_valid) = 0;
};

}  // namespace grassland::rhi::backend
