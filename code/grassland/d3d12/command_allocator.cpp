#include "grassland/d3d12/command_allocator.h"

#include "grassland/d3d12/command_list.h"

namespace grassland::d3d12 {

CommandAllocator::CommandAllocator(
    const ComPtr<ID3D12CommandAllocator> &command_allocator)
    : command_allocator_(command_allocator) {
}

HRESULT CommandAllocator::CreateCommandList(
    D3D12_COMMAND_LIST_TYPE type,
    double_ptr<CommandList> pp_command_list) {
  ComPtr<ID3D12Device> device;
  RETURN_IF_FAILED_HR(command_allocator_->GetDevice(IID_PPV_ARGS(&device)),
                      "Failed to get device from command allocator");

  ComPtr<ID3D12GraphicsCommandList> command_list;
  device->CreateCommandList(0, type, command_allocator_.Get(), nullptr,
                            IID_PPV_ARGS(&command_list));

  command_list->Close();

  pp_command_list.construct(command_list);

  return S_OK;
}

HRESULT CommandAllocator::ResetCommandRecord(CommandList *command_list) {
  RETURN_IF_FAILED_HR(command_allocator_->Reset(),
                      "Failed to reset command allocator");
  RETURN_IF_FAILED_HR(
      command_list->Handle()->Reset(command_allocator_.Get(), nullptr),
      "Failed to reset command list");

  return S_OK;
}

}  // namespace grassland::d3d12
