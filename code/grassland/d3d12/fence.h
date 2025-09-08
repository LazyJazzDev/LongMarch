#pragma once
#include "grassland/d3d12/command_queue.h"
#include "grassland/d3d12/device.h"

namespace CD::d3d12 {

class Fence {
 public:
  Fence(const ComPtr<ID3D12Fence> &fence);

  ID3D12Fence *Handle() const {
    return fence_.Get();
  }

  // Last value that was signaled.
  uint64_t Value() const {
    return value_;
  }

  HRESULT Signal(CommandQueue *command_queue);

  HRESULT Wait(CommandQueue *command_queue);

  HRESULT WaitFor(uint64_t value);

  HRESULT Wait();

  // This function is designed for cross API external signaling like with CUDA.
  // Simply updating this value could cause issues if the fence is not synchronized properly with the GPU.
  // Don't call this function unless you know what you're doing.
  void ExternalSignalUpdateValue(uint64_t value) {
    value_ = value;
  }

 private:
  ComPtr<ID3D12Fence> fence_;
  HANDLE fence_event_;
  uint64_t value_;
};

}  // namespace CD::d3d12
