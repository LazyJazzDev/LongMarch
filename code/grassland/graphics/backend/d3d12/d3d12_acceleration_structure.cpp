#include "grassland/graphics/backend/d3d12/d3d12_acceleration_structure.h"

#include "grassland/graphics/backend/d3d12/d3d12_core.h"

namespace grassland::graphics::backend {

D3D12AccelerationStructure::D3D12AccelerationStructure(
    D3D12Core *core,
    std::unique_ptr<d3d12::AccelerationStructure> &&acceleration_structure)
    : core_(core), acceleration_structure_(std::move(acceleration_structure)) {
}

int D3D12AccelerationStructure::UpdateInstances(
    const std::vector<std::pair<AccelerationStructure *, glm::mat4>> &instances) {
  std::vector<std::pair<d3d12::AccelerationStructure *, glm::mat4>> d3d12_instances;
  d3d12_instances.reserve(instances.size());
  for (const auto &instance : instances) {
    D3D12AccelerationStructure *d3d12_instance = dynamic_cast<D3D12AccelerationStructure *>(instance.first);
    assert(d3d12_instance != nullptr);
    d3d12_instances.push_back({d3d12_instance->acceleration_structure_.get(), instance.second});
  }
  acceleration_structure_->UpdateInstances(d3d12_instances, core_->CommandQueue(), core_->SingleTimeFence(),
                                           core_->SingleTimeCommandAllocator());
  return 0;
}

}  // namespace grassland::graphics::backend
