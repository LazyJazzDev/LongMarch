#include "cao_di/d3d12/raytracing/raytracing_pipeline.h"

namespace CD::d3d12 {

RayTracingPipeline::RayTracingPipeline(const ComPtr<ID3D12StateObject> &state_object,
                                       size_t miss_shader_count,
                                       size_t hit_group_count,
                                       size_t callable_shader_count)
    : state_object_(state_object),
      miss_shader_count_(miss_shader_count),
      hit_group_count_(hit_group_count),
      callable_shader_count_(callable_shader_count) {
}

}  // namespace CD::d3d12
