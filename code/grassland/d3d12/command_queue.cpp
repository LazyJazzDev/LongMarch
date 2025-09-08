#include "grassland/d3d12/command_queue.h"

#include "grassland/d3d12/command_allocator.h"
#include "grassland/d3d12/fence.h"

namespace CD::d3d12 {

CommandQueue::CommandQueue(const ComPtr<ID3D12CommandQueue> &command_queue) : command_queue_(command_queue) {
}

HRESULT CommandQueue::SingleTimeCommand(Fence *fence,
                                        CommandAllocator *command_allocator,
                                        const std::function<void(ID3D12GraphicsCommandList *)> &function) {
  ComPtr<ID3D12Device> device;
  RETURN_IF_FAILED_HR(command_queue_->GetDevice(IID_PPV_ARGS(&device)), "failed to get device.");

  RETURN_IF_FAILED_HR(command_allocator->Handle()->Reset(), "failed to reset command allocator.");

  ComPtr<ID3D12GraphicsCommandList> command_list;
  RETURN_IF_FAILED_HR(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, command_allocator->Handle(), nullptr,
                                                IID_PPV_ARGS(&command_list)),
                      "failed to create command list.");

  function(command_list.Get());

  command_list->Close();

  ID3D12CommandList *command_lists[] = {command_list.Get()};

  command_queue_->ExecuteCommandLists(1, command_lists);

  RETURN_IF_FAILED_HR(fence->Signal(this), "failed to signal fence.");

  fence->Wait();

  return S_OK;
}

HRESULT CommandQueue::SingleTimeCommand(Fence *fence,
                                        const std::function<void(ID3D12GraphicsCommandList *)> &function) {
  ComPtr<ID3D12Device> device;
  RETURN_IF_FAILED_HR(command_queue_->GetDevice(IID_PPV_ARGS(&device)), "failed to get device.");

  ComPtr<ID3D12CommandAllocator> command_allocator;
  RETURN_IF_FAILED_HR(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&command_allocator)),
                      "failed to create command allocator.");

  CommandAllocator command_allocator_wrapper(command_allocator);

  return SingleTimeCommand(fence, &command_allocator_wrapper, function);
}

HRESULT CommandQueue::SingleTimeCommand(const std::function<void(ID3D12GraphicsCommandList *)> &function) {
  ComPtr<ID3D12Device> device;
  RETURN_IF_FAILED_HR(command_queue_->GetDevice(IID_PPV_ARGS(&device)), "failed to get device.");

  ComPtr<ID3D12Fence> fence;
  RETURN_IF_FAILED_HR(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "failed to create fence.");

  Fence fence_wrapper(fence);

  return SingleTimeCommand(&fence_wrapper, function);
}

}  // namespace CD::d3d12
