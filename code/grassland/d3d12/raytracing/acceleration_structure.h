#pragma once
#include "grassland/d3d12/command_allocator.h"
#include "grassland/d3d12/d3d12util.h"

namespace grassland::d3d12 {

class AccelerationStructure {
 public:
  AccelerationStructure(const ComPtr<ID3D12Resource> &as);

  ID3D12Resource *Handle() const {
    return as_.Get();
  }

 private:
  ComPtr<ID3D12Resource> as_;
};

}  // namespace grassland::d3d12
