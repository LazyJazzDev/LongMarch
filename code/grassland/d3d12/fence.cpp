#include "grassland/d3d12/fence.h"

namespace grassland::d3d12 {

Fence::Fence(const ComPtr<ID3D12Fence> &fence) : fence_(fence), value_(1) {
  fence_event_ = CreateEvent(nullptr, FALSE, FALSE, nullptr);
  if (fence_event_ == nullptr) {
    throw std::runtime_error("failed to create fence event.");
  }
}

HRESULT Fence::Signal(CommandQueue *command_queue) {
  value_++;
  RETURN_IF_FAILED_HR(command_queue->Handle()->Signal(fence_.Get(), value_), "failed to signal fence.");
  return S_OK;
}

HRESULT Fence::Wait(CommandQueue *command_queue) {
  RETURN_IF_FAILED_HR(command_queue->Handle()->Wait(fence_.Get(), value_), "failed to wait fence.");
  return S_OK;
}

HRESULT Fence::WaitFor(uint64_t value) {
  if (fence_->GetCompletedValue() < value) {
    RETURN_IF_FAILED_HR(fence_->SetEventOnCompletion(value, fence_event_), "failed to set event on completion.");
    WaitForSingleObject(fence_event_, INFINITE);
  }
  return S_OK;
}

HRESULT Fence::Wait() {
  return WaitFor(value_);
}

}  // namespace grassland::d3d12
