#include "grassland/rhi/backend/d3d12/d3d12_queue.h"

#include "grassland/rhi/backend/d3d12/d3d12_command_context.h"
#include "grassland/rhi/backend/d3d12/d3d12_device.h"

namespace grassland::rhi::backend::d3d12 {

rhi::ICommandContext *D3D12Queue::CreateCommandContext() {
  return D3D12CommandContext::Create(this);
}

D3D12Queue::D3D12Queue(D3D12Device *device, QueueType type) : Queue(device, type) {
}

}  // namespace grassland::rhi::backend::d3d12
