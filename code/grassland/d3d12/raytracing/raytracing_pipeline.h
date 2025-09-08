#pragma once
#include "grassland/d3d12/d3d12util.h"

namespace CD::d3d12 {

class RayTracingPipeline {
 public:
  RayTracingPipeline(const ComPtr<ID3D12StateObject> &state_object,
                     size_t miss_shader_count,
                     size_t hit_group_count,
                     size_t callable_shader_count);
  ~RayTracingPipeline() = default;

  ID3D12StateObject *Handle() const {
    return state_object_.Get();
  }

  size_t MissShaderCount() const {
    return miss_shader_count_;
  }

  size_t HitGroupCount() const {
    return hit_group_count_;
  }

  size_t CallableShaderCount() const {
    return callable_shader_count_;
  }

 private:
  ComPtr<ID3D12StateObject> state_object_;
  size_t miss_shader_count_{0};
  size_t hit_group_count_{0};
  size_t callable_shader_count_{0};
};

}  // namespace CD::d3d12
