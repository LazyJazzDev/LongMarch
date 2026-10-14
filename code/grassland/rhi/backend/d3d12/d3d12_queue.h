#pragma once
#include "grassland/rhi/backend/backend_queue.h"
#include "grassland/rhi/backend/d3d12/d3d12_device.h"

namespace grassland::rhi::backend::d3d12 {

class D3D12Queue final : public IQueue {
 public:
  D3D12Queue(const ObjectPtr<D3D12Device> &device, QueueType type);
  std::unique_ptr<ICommandContext> CreateCommandContext() const override;
};

}  // namespace grassland::rhi::backend::d3d12
