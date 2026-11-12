#include "grassland/rhi/backend/d3d12/d3d12_timeline.h"

#include "grassland/rhi/backend/d3d12/d3d12_device.h"

namespace grassland::rhi::backend::d3d12 {

D3D12TimelineImpl::D3D12TimelineImpl(ID3D12Device *device) {
  InitializeTimeline(device);
}

ObjectPtr<ID3D12Timeline> D3D12TimelineImpl::Create(ID3D12Device *device) {
  auto result = ObjectPtr<D3D12TimelineImpl>::Adopt(new D3D12TimelineImpl(device));
  if (!Check(device->NativeDevice()->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&result->fence_)),
             "Create queue timeline"))
    return {};
  return result;
}

bool D3D12TimelineImpl::GetCompletedValue(uint64_t &value) const {
  value = fence_->GetCompletedValue();
  return value != UINT64_MAX;
}

bool D3D12TimelineImpl::WaitHost(uint64_t value) const {
  uint64_t completed;
  if (!GetCompletedValue(completed))
    return false;
  if (completed >= value)
    return true;
  // A null event requests the native blocking wait without a separate Win32 handle.
  return Check(fence_->SetEventOnCompletion(value, nullptr), "Wait for timeline") && GetCompletedValue(completed) &&
         completed >= value;
}

ID3D12Fence *D3D12TimelineImpl::NativeFence() const {
  return fence_.Get();
}

ObjectPtr<ID3D12Timeline> ID3D12Timeline::Create(ID3D12Device *device) {
  return D3D12TimelineImpl::Create(device);
}
}  // namespace grassland::rhi::backend::d3d12
