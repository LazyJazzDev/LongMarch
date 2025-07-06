#include "grassland/d3d12/raytracing/raytracing_pipeline.h"

namespace grassland::d3d12 {

RayTracingPipeline::RayTracingPipeline(const ComPtr<ID3D12StateObject> &state_object) : state_object_(state_object) {
}

}  // namespace grassland::d3d12
