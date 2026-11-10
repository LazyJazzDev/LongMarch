#include "grassland/rhi/backend/d3d12/d3d12_queue.h"

#include "grassland/rhi/backend/d3d12/d3d12_command_context.h"
#include "grassland/rhi/backend/d3d12/d3d12_device.h"

namespace grassland::rhi::backend::d3d12 {

rhi::ICommandContext *D3D12Queue::CreateCommandContext() {
  return D3D12CommandContext::Create(this);
}

D3D12Queue *D3D12Queue::Create(D3D12Device *device, QueueType type) {
  auto result = std::unique_ptr<D3D12Queue>(new D3D12Queue(device, type));
  D3D12_COMMAND_QUEUE_DESC desc{};
  desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  if (!Check(device->NativeDevice()->CreateCommandQueue(&desc, IID_PPV_ARGS(&result->queue_)), "Create command queue"))
    return nullptr;
  result->timeline_ = D3D12Timeline::Create(device);
  if (!result->timeline_)
    return nullptr;
  return result.release();
}

ID3D12CommandQueue *D3D12Queue::NativeQueue() const {
  return queue_.Get();
}

D3D12Timeline *D3D12Queue::GetTimeline() const {
  return timeline_.get();
}

bool D3D12Queue::WaitIdle() {
  const auto value = NextSerial();
  return Check(queue_->Signal(timeline_->NativeFence(), value), "Signal idle timeline") && timeline_->WaitHost(value);
}

D3D12Queue::D3D12Queue(D3D12Device *device, QueueType type) : Queue(device, type) {
}

}  // namespace grassland::rhi::backend::d3d12
