#include "grassland/graphics/backend/d3d12/d3d12_acceleration_structure.h"

#include "grassland/graphics/backend/d3d12/d3d12_core.h"

namespace grassland::graphics::backend {

D3D12AccelerationStructure::D3D12AccelerationStructure(
    D3D12Core *core,
    std::unique_ptr<d3d12::AccelerationStructure> &&acceleration_structure)
    : core_(core), acceleration_structure_(std::move(acceleration_structure)) {
}

int D3D12AccelerationStructure::UpdateInstances(const std::vector<RayTracingInstance> &instances) {
  std::vector<D3D12_RAYTRACING_INSTANCE_DESC> d3d12_instances;
  d3d12_instances.reserve(instances.size());
  for (const auto &instance : instances) {
    d3d12_instances.emplace_back(RayTracingInstanceToD3D12RayTracingInstanceDesc(instance));
  }
  acceleration_structure_->UpdateInstances(d3d12_instances, core_->CommandQueue(), core_->SingleTimeFence(),
                                           core_->SingleTimeCommandAllocator());
  return 0;
}

}  // namespace grassland::graphics::backend
