#include "grassland/rhi/backend/d3d12/d3d12_queue.h"

#include "grassland/rhi/backend/d3d12/d3d12_command_context.h"
#include "grassland/rhi/backend/d3d12/d3d12_device.h"

namespace grassland::rhi::backend::d3d12 {
D3D12Queue::D3D12Queue(D3D12Device *device, QueueType type) : Queue(device, type) {
}

D3D12Queue *D3D12Queue::Create(D3D12Device *device, QueueType type) {
  auto result = std::unique_ptr<D3D12Queue>(new D3D12Queue(device, type));
  D3D12_COMMAND_QUEUE_DESC desc{};
  desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  if (!Check(device->NativeDevice()->CreateCommandQueue(&desc, IID_PPV_ARGS(&result->queue_)),
             "Create command queue") ||
      !Check(device->NativeDevice()->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&result->timeline_)),
             "Create queue timeline"))
    return nullptr;
  return result.release();
}

ID3D12CommandQueue *D3D12Queue::Native() const {
  return queue_.Get();
}

ID3D12Fence *D3D12Queue::Timeline() const {
  return timeline_.Get();
}

const void *D3D12Queue::NativeIdentity() const {
  return queue_.Get();
}

bool D3D12Queue::QueryTimeline(uint64_t &value) const {
  value = timeline_->GetCompletedValue();
  return value != UINT64_MAX;
}

bool D3D12Queue::Wait(uint64_t value, HANDLE event) const {
  auto fence = Timeline();
  if (fence->GetCompletedValue() == UINT64_MAX)
    return false;
  if (fence->GetCompletedValue() >= value)
    return true;
  if (!Check(fence->SetEventOnCompletion(value, event), "SetEventOnCompletion"))
    return false;
  DWORD result;
  do {
    result = WaitForSingleObject(event, 50);
    if (FAILED(static_cast<D3D12Device *>(Owner())->NativeDevice()->GetDeviceRemovedReason()))
      return false;
  } while (result == WAIT_TIMEOUT);
  return result == WAIT_OBJECT_0 && fence->GetCompletedValue() != UINT64_MAX;
}

bool D3D12Queue::WaitIdle() {
  const auto value = NextSerial();
  return Check(Native()->Signal(Timeline(), value), "Signal idle fence") &&
         Owner()->WaitTimelines({{this, value}}, true);
}

ICommandContext *D3D12Queue::CreateCommandContext() {
  return D3D12CommandContext::Create(this);
}

}  // namespace grassland::rhi::backend::d3d12
