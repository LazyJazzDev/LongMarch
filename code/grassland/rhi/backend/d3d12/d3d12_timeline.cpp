#include "grassland/rhi/backend/d3d12/d3d12_timeline.h"

#include "grassland/rhi/backend/d3d12/d3d12_device.h"

namespace grassland::rhi::backend::d3d12 {

D3D12Timeline::D3D12Timeline(D3D12Device *device) : Timeline(device) {
}

std::unique_ptr<D3D12Timeline> D3D12Timeline::Create(D3D12Device *device) {
  auto result = std::unique_ptr<D3D12Timeline>(new D3D12Timeline(device));
  if (!Check(device->NativeDevice()->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&result->fence_)),
             "Create queue timeline"))
    return {};
  return result;
}

bool D3D12Timeline::GetCompletedValue(uint64_t &value) const {
  value = fence_->GetCompletedValue();
  return value != UINT64_MAX;
}

bool D3D12Timeline::WaitHost(uint64_t value) const {
  uint64_t completed;
  if (!GetCompletedValue(completed))
    return false;
  if (completed >= value)
    return true;
  // A null event requests the native blocking wait without a separate Win32 handle.
  return Check(fence_->SetEventOnCompletion(value, nullptr), "Wait for timeline") && GetCompletedValue(completed) &&
         completed >= value;
}

ID3D12Fence *D3D12Timeline::NativeFence() const {
  return fence_.Get();
}

}  // namespace grassland::rhi::backend::d3d12
