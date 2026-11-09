#pragma once
#include "grassland/rhi/backend/backend_queue.h"
#include "grassland/rhi/backend/metal/metal_device.h"
#include "grassland/rhi/backend/metal/metal_submission_token.h"

namespace grassland::rhi::backend::metal {

class MetalQueue final : public Queue {
 public:
  rhi::ISubmissionToken *Submit(rhi::ICommandContext *commands, const rhi::SubmitInfo &info = {}) override {
    return SubmitCommands(commands, info,
                          ObjectPtr<backend::SubmissionToken>::Adopt(new MetalSubmissionToken(Owner())));
  }

  QueueType GetType() const override {
    return QueueRole();
  }

  bool SharesPhysicalQueue(const rhi::IQueue *other) const override {
    return UsesSameDeviceQueue(other);
  }

  MetalQueue(MetalDevice *device, QueueType type);
};

}  // namespace grassland::rhi::backend::metal
