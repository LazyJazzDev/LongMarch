#include "grassland/rhi/backend/d3d12/d3d12_queue.h"

#include "grassland/rhi/backend/d3d12/d3d12_command_context.h"
#include "grassland/rhi/backend/d3d12/d3d12_device.h"

namespace grassland::rhi::backend::d3d12 {
D3D12Queue::D3D12Queue(D3D12Device *device, QueueType type, ObjectPtr<D3D12Timeline> timeline)
    : Queue(device, type, std::move(timeline)) {
}

D3D12Queue *D3D12Queue::Create(D3D12Device *device, QueueType type) {
  auto timeline = ObjectPtr<D3D12Timeline>::Adopt(D3D12Timeline::Create(device));
  if (!timeline)
    return nullptr;
  auto result = std::unique_ptr<D3D12Queue>(new D3D12Queue(device, type, std::move(timeline)));
  D3D12_COMMAND_QUEUE_DESC desc{};
  desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  if (!Check(device->NativeDevice()->CreateCommandQueue(&desc, IID_PPV_ARGS(&result->queue_)), "Create command queue"))
    return nullptr;
  return result.release();
}

ID3D12CommandQueue *D3D12Queue::Native() const {
  return queue_.Get();
}

ID3D12Fence *D3D12Queue::Timeline() const {
  return static_cast<D3D12Timeline *>(TimelineState())->Native();
}

const void *D3D12Queue::NativeIdentity() const {
  return queue_.Get();
}

bool D3D12Queue::Wait(uint64_t value, HANDLE event) const {
  return static_cast<D3D12Timeline *>(TimelineState())->Wait(value, event);
}

bool D3D12Queue::WaitIdle() {
  const auto value = NextSerial();
  return Check(Native()->Signal(Timeline(), value), "Signal idle fence") && TimelineState()->WaitHost(value);
}

ICommandContext *D3D12Queue::CreateCommandContext() {
  return D3D12CommandContext::Create(this);
}

}  // namespace grassland::rhi::backend::d3d12
