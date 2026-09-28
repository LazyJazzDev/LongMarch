#pragma once
#include "grassland/graphics/backend/d3d12/helper/device.h"

namespace grassland::graphics::backend::d3d12 {

class CommandAllocator {
 public:
  CommandAllocator(const ComPtr<ID3D12CommandAllocator> &command_allocator);

  ID3D12CommandAllocator *Handle() const {
    return command_allocator_.Get();
  }

  HRESULT CreateCommandList(D3D12_COMMAND_LIST_TYPE type, double_ptr<CommandList> pp_command_list);

  HRESULT ResetCommandRecord(CommandList *command_list);

 private:
  ComPtr<ID3D12CommandAllocator> command_allocator_;
};

}  // namespace grassland::graphics::backend::d3d12
