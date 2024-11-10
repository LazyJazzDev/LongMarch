#pragma once
#include "grassland/d3d12/command_queue.h"
#include "grassland/d3d12/device.h"

namespace grassland::d3d12 {

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

  HRESULT WaitFor(uint64_t value);

  HRESULT Wait();

 private:
  ComPtr<ID3D12Fence> fence_;
  HANDLE fence_event_;
  uint64_t value_;
};

}  // namespace grassland::d3d12
