#pragma once
#include "grassland/rhi/backend/backend_command_context.h"
#include "grassland/rhi/backend/backend_device.h"
#include "grassland/rhi/rhi_queue.h"

namespace grassland::rhi::backend {

class Queue : public Object<rhi::IQueue> {
 public:
  Queue(Device *device, QueueType type);
  ~Queue() override;
  Device *Owner() const;
  QueueType Type() const;
  QueueType GetType() const override;
  bool SharesPhysicalQueue(const rhi::IQueue *other) const override;
  rhi::ISubmissionToken *Submit(rhi::ICommandContext *commands, const rhi::SubmitInfo &info = {}) override;

 private:
  Device *device_;
  QueueType type_;
};

}  // namespace grassland::rhi::backend
