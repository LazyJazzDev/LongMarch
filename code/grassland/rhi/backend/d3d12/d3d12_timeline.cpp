#include "grassland/rhi/backend/d3d12/d3d12_timeline.h"

#include "grassland/rhi/backend/d3d12/d3d12_device.h"

namespace grassland::rhi::backend::d3d12 {
D3D12Timeline::D3D12Timeline(D3D12Device *device) : backend::Timeline(device) {
}

D3D12Timeline *D3D12Timeline::Create(D3D12Device *device) {
  auto result = std::unique_ptr<D3D12Timeline>(new D3D12Timeline(device));
  if (!Check(device->NativeDevice()->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&result->fence_)),
             "Create queue timeline"))
    return nullptr;
  return result.release();
}

D3D12Timeline::~D3D12Timeline() = default;

ID3D12Fence *D3D12Timeline::Native() const {
  return fence_.Get();
}

bool D3D12Timeline::GetCompletedValue(uint64_t &value) const {
  value = fence_->GetCompletedValue();
  return value != UINT64_MAX;
}

bool D3D12Timeline::Wait(uint64_t value, HANDLE event) const {
  uint64_t completed{};
  if (!GetCompletedValue(completed))
    return false;
  if (completed >= value)
    return true;
  if (!Check(fence_->SetEventOnCompletion(value, event), "SetEventOnCompletion"))
    return false;
  DWORD result;
  do {
    result = WaitForSingleObject(event, 50);
    if (FAILED(static_cast<D3D12Device *>(Owner())->NativeDevice()->GetDeviceRemovedReason()))
      return false;
  } while (result == WAIT_TIMEOUT);
  return result == WAIT_OBJECT_0 && fence_->GetCompletedValue() != UINT64_MAX;
}

}  // namespace grassland::rhi::backend::d3d12
