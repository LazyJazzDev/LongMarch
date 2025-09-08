#include "grassland/d3d12/raytracing/shader_table.h"

namespace CD::d3d12 {
ShaderTable::ShaderTable(const ComPtr<ID3D12Resource> &buffer,
                         D3D12_GPU_VIRTUAL_ADDRESS ray_gen_offset,
                         D3D12_GPU_VIRTUAL_ADDRESS miss_offset,
                         D3D12_GPU_VIRTUAL_ADDRESS hit_group_offset,
                         D3D12_GPU_VIRTUAL_ADDRESS callable_offset,
                         size_t miss_shader_count,
                         size_t hit_group_count,
                         size_t callable_count)
    : buffer_(buffer),
      ray_gen_offset_(ray_gen_offset),
      miss_offset_(miss_offset),
      hit_group_offset_(hit_group_offset),
      callable_offset_(callable_offset),
      miss_shader_count_(miss_shader_count),
      hit_group_count_(hit_group_count),
      callable_count_(callable_count) {
}

D3D12_GPU_VIRTUAL_ADDRESS ShaderTable::GetRayGenDeviceAddress() const {
  return buffer_->GetGPUVirtualAddress() + ray_gen_offset_;
}

D3D12_GPU_VIRTUAL_ADDRESS ShaderTable::GetMissDeviceAddress() const {
  return buffer_->GetGPUVirtualAddress() + miss_offset_;
}

D3D12_GPU_VIRTUAL_ADDRESS ShaderTable::GetHitGroupDeviceAddress() const {
  return buffer_->GetGPUVirtualAddress() + hit_group_offset_;
}

D3D12_GPU_VIRTUAL_ADDRESS ShaderTable::GetCallableDeviceAddress() const {
  return buffer_->GetGPUVirtualAddress() + callable_offset_;
}

size_t ShaderTable::MissShaderCount() const {
  return miss_shader_count_;
}

size_t ShaderTable::HitGroupShaderCount() const {
  return hit_group_count_;
}

size_t ShaderTable::CallableShaderCount() const {
  return callable_count_;
}

}  // namespace CD::d3d12
