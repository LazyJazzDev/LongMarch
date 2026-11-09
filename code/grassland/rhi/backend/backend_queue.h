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
  QueueType QueueRole() const;
  bool UsesSameDeviceQueue(const rhi::IQueue *other) const;
  rhi::ISubmissionToken *SubmitCommands(rhi::ICommandContext *commands,
                                        const rhi::SubmitInfo &info,
                                        ObjectPtr<SubmissionToken> completion);

 private:
  Device *device_;
  QueueType type_;
};

}  // namespace grassland::rhi::backend
