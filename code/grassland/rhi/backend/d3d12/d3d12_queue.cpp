#include "grassland/rhi/backend/d3d12/d3d12_queue.h"

#include "grassland/rhi/backend/d3d12/d3d12_command_context.h"
#include "grassland/rhi/backend/d3d12/d3d12_command_context_impl.h"
#include "grassland/rhi/backend/d3d12/d3d12_device.h"
#include "grassland/rhi/backend/d3d12/d3d12_queue_impl.h"
#include "grassland/rhi/backend/d3d12/d3d12_timeline_impl.h"

namespace grassland::rhi::backend::d3d12 {

rhi::ICommandContext *D3D12QueueImpl::CreateCommandContext() {
  return ID3D12CommandContext::Create(this);
}

ID3D12Queue *D3D12QueueImpl::Create(ID3D12Device *device, QueueType type) {
  auto result = std::unique_ptr<D3D12QueueImpl>(new D3D12QueueImpl(device, type));
  D3D12_COMMAND_QUEUE_DESC desc{};
  desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  if (!Check(device->NativeDevice()->CreateCommandQueue(&desc, IID_PPV_ARGS(&result->queue_)), "Create command queue"))
    return nullptr;
  result->timeline_ = ID3D12Timeline::Create(device);
  if (!result->timeline_)
    return nullptr;
  return result.release();
}

ID3D12CommandQueue *D3D12QueueImpl::NativeQueue() const {
  return queue_.Get();
}

ID3D12Timeline *D3D12QueueImpl::GetTimeline() const {
  return timeline_.Get();
}

bool D3D12QueueImpl::WaitIdle() {
  const auto value = NextSerial();
  return Check(queue_->Signal(timeline_->NativeFence(), value), "Signal idle timeline") && timeline_->WaitHost(value);
}

D3D12QueueImpl::D3D12QueueImpl(ID3D12Device *device, QueueType type) {
  InitializeQueue(device, type);
}

}  // namespace grassland::rhi::backend::d3d12

LONGMARCH_RHI_UUID(grassland::rhi::backend::d3d12::ID3D12Queue, "46194ddb-7a1a-4d51-b878-0e51d212d642")

namespace grassland::rhi::backend::d3d12 {
ID3D12Queue *ID3D12Queue::Create(ID3D12Device *device, QueueType type) {
  return D3D12QueueImpl::Create(device, type);
}
}  // namespace grassland::rhi::backend::d3d12
