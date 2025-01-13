#pragma once
#include "grassland/d3d12/d3d12util.h"

namespace grassland::d3d12 {

class ShaderTable {
 public:
  ShaderTable(const ComPtr<ID3D12Resource> &buffer,
              D3D12_GPU_VIRTUAL_ADDRESS ray_gen_offset,
              D3D12_GPU_VIRTUAL_ADDRESS miss_offset,
              D3D12_GPU_VIRTUAL_ADDRESS hit_group_offset);
  D3D12_GPU_VIRTUAL_ADDRESS GetRayGenDeviceAddress() const;
  D3D12_GPU_VIRTUAL_ADDRESS GetMissDeviceAddress() const;
  D3D12_GPU_VIRTUAL_ADDRESS GetHitGroupDeviceAddress() const;

 private:
  ComPtr<ID3D12Resource> buffer_;
  D3D12_GPU_VIRTUAL_ADDRESS ray_gen_offset_;
  D3D12_GPU_VIRTUAL_ADDRESS miss_offset_;
  D3D12_GPU_VIRTUAL_ADDRESS hit_group_offset_;
};

}  // namespace grassland::d3d12
