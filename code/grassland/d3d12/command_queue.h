#pragma once
#include "grassland/d3d12/device.h"

namespace grassland::d3d12 {

class CommandQueue {
 public:
  CommandQueue(const ComPtr<ID3D12CommandQueue> &command_queue);

  ID3D12CommandQueue *Handle() const {
    return command_queue_.Get();
  }

 private:
  ComPtr<ID3D12CommandQueue> command_queue_;
};

}  // namespace grassland::d3d12
