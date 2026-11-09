#pragma once
#include "grassland/rhi/backend/backend_queue.h"
#include "grassland/rhi/backend/d3d12/d3d12_device.h"
#include "grassland/rhi/backend/d3d12/d3d12_submission_token.h"

namespace grassland::rhi::backend::d3d12 {

class D3D12Queue final : public Queue {
 public:
  rhi::ISubmissionToken *Submit(rhi::ICommandContext *commands, const rhi::SubmitInfo &info = {}) override {
    return SubmitCommands(commands, info,
                          ObjectPtr<backend::SubmissionToken>::Adopt(new D3D12SubmissionToken(Owner())));
  }

  QueueType GetType() const override {
    return QueueRole();
  }

  bool SharesPhysicalQueue(const rhi::IQueue *other) const override {
    return UsesSameDeviceQueue(other);
  }

  D3D12Queue(D3D12Device *device, QueueType type);
};

}  // namespace grassland::rhi::backend::d3d12
