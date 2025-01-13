#pragma once
#include "grassland/d3d12/d3d12util.h"

namespace grassland::d3d12 {

class RayTracingPipeline {
 public:
  RayTracingPipeline(const ComPtr<ID3D12StateObject> &state_object,
                     const std::wstring &ray_gen_shader_name,
                     const std::wstring &miss_shader_name,
                     const std::wstring &hit_group_name);
  ~RayTracingPipeline() = default;

  ID3D12StateObject *Handle() const {
    return state_object_.Get();
  }

  std::wstring RayGenShaderName() const {
    return ray_gen_shader_name_;
  }

  std::wstring MissShaderName() const {
    return miss_shader_name_;
  }

  std::wstring HitGroupName() const {
    return hit_group_name_;
  }

 private:
  ComPtr<ID3D12StateObject> state_object_;
  std::wstring ray_gen_shader_name_;
  std::wstring miss_shader_name_;
  std::wstring hit_group_name_;
};

}  // namespace grassland::d3d12
