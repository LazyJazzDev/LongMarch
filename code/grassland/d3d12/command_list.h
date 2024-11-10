#pragma once
#include "grassland/d3d12/device.h"

namespace grassland::d3d12 {

class CommandList {
 public:
  CommandList(const ComPtr<ID3D12GraphicsCommandList> &command_list);

  ID3D12GraphicsCommandList *Handle() const {
    return command_list_.Get();
  }

 private:
  ComPtr<ID3D12GraphicsCommandList> command_list_;
};

}  // namespace grassland::d3d12
