#include "grassland/rhi/backend/d3d12/d3d12_queue.h"

#include "grassland/rhi/backend/d3d12/d3d12_command_context.h"

namespace grassland::rhi::backend::d3d12 {

D3D12Queue::D3D12Queue(D3D12Device *device, QueueType type) : IQueue(device, type) {
}

std::unique_ptr<ICommandContext> D3D12Queue::CreateCommandContext() const {
  return std::make_unique<D3D12CommandContext>(static_cast<D3D12Device *>(Owner()), Type());
}

}  // namespace grassland::rhi::backend::d3d12
