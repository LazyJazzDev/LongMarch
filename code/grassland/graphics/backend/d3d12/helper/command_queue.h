#pragma once
#include "grassland/graphics/backend/d3d12/helper/command_list.h"
#include "grassland/graphics/backend/d3d12/helper/device.h"

namespace grassland::graphics::backend::d3d12 {

class CommandQueue {
 public:
  CommandQueue(const ComPtr<ID3D12CommandQueue> &command_queue);

  ID3D12CommandQueue *Handle() const {
    return command_queue_.Get();
  }

  HRESULT SingleTimeCommand(Fence *fence,
                            CommandAllocator *command_allocator,
                            const std::function<void(ID3D12GraphicsCommandList *)> &function);

  HRESULT SingleTimeCommand(Fence *fence, const std::function<void(ID3D12GraphicsCommandList *)> &function);

  HRESULT SingleTimeCommand(const std::function<void(ID3D12GraphicsCommandList *)> &function);

 private:
  ComPtr<ID3D12CommandQueue> command_queue_;
};

}  // namespace grassland::graphics::backend::d3d12
