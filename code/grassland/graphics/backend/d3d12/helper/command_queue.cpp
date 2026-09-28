#include "grassland/graphics/backend/d3d12/helper/command_queue.h"

namespace grassland::graphics::backend::d3d12 {
HRESULT SingleTimeCommand(ID3D12CommandQueue *queue,
                          ID3D12CommandAllocator *allocator,
                          const std::function<void(ID3D12GraphicsCommandList *)> &function) {
  ComPtr<ID3D12Device> device;
  RETURN_IF_FAILED_HR(queue->GetDevice(IID_PPV_ARGS(device.GetAddressOf())), "Failed to get queue device");
  RETURN_IF_FAILED_HR(allocator->Reset(), "Failed to reset command allocator");
  ComPtr<ID3D12GraphicsCommandList> list;
  RETURN_IF_FAILED_HR(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, nullptr,
                                                IID_PPV_ARGS(list.GetAddressOf())),
                      "Failed to create command list");
  function(list.Get());
  RETURN_IF_FAILED_HR(list->Close(), "Failed to close command list");
  ID3D12CommandList *lists[] = {list.Get()};
  queue->ExecuteCommandLists(1, lists);
  ComPtr<ID3D12Fence> fence;
  RETURN_IF_FAILED_HR(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(fence.GetAddressOf())),
                      "Failed to create command fence");
  RETURN_IF_FAILED_HR(queue->Signal(fence.Get(), 1), "Failed to signal command fence");
  HANDLE event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
  if (!event) {
    return HRESULT_FROM_WIN32(GetLastError());
  }
  HRESULT result = fence->SetEventOnCompletion(1, event);
  if (SUCCEEDED(result)) {
    WaitForSingleObject(event, INFINITE);
  }
  CloseHandle(event);
  return result;
}
}  // namespace grassland::graphics::backend::d3d12
