#pragma once
#include "cao_di/d3d12/d3d12util.h"

namespace CD::d3d12 {

class ShaderTable {
 public:
  ShaderTable(const ComPtr<ID3D12Resource> &buffer,
              D3D12_GPU_VIRTUAL_ADDRESS ray_gen_offset,
              D3D12_GPU_VIRTUAL_ADDRESS miss_offset,
              D3D12_GPU_VIRTUAL_ADDRESS hit_group_offset,
              D3D12_GPU_VIRTUAL_ADDRESS callable_offset,
              size_t miss_shader_count,
              size_t hit_group_count,
              size_t callable_count);
  D3D12_GPU_VIRTUAL_ADDRESS GetRayGenDeviceAddress() const;
  D3D12_GPU_VIRTUAL_ADDRESS GetMissDeviceAddress() const;
  D3D12_GPU_VIRTUAL_ADDRESS GetHitGroupDeviceAddress() const;
  D3D12_GPU_VIRTUAL_ADDRESS GetCallableDeviceAddress() const;

  size_t MissShaderCount() const;
  size_t HitGroupShaderCount() const;
  size_t CallableShaderCount() const;

 private:
  ComPtr<ID3D12Resource> buffer_;
  D3D12_GPU_VIRTUAL_ADDRESS ray_gen_offset_;
  D3D12_GPU_VIRTUAL_ADDRESS miss_offset_;
  D3D12_GPU_VIRTUAL_ADDRESS hit_group_offset_;
  D3D12_GPU_VIRTUAL_ADDRESS callable_offset_;
  size_t miss_shader_count_{0};
  size_t hit_group_count_{0};
  size_t callable_count_{0};
};

}  // namespace CD::d3d12
