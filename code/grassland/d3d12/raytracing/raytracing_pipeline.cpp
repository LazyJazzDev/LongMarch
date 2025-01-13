#include "grassland/d3d12/raytracing/raytracing_pipeline.h"

namespace grassland::d3d12 {

RayTracingPipeline::RayTracingPipeline(const ComPtr<ID3D12StateObject> &state_object,
                                       const std::wstring &ray_gen_shader_name,
                                       const std::wstring &miss_shader_name,
                                       const std::wstring &hit_group_name)
    : state_object_(state_object),
      ray_gen_shader_name_(ray_gen_shader_name),
      miss_shader_name_(miss_shader_name),
      hit_group_name_(hit_group_name) {
}

}  // namespace grassland::d3d12
