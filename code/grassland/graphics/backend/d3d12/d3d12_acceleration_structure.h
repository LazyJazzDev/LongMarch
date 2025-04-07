#pragma once
#include "grassland/graphics/backend/d3d12/d3d12_util.h"

namespace grassland::graphics::backend {

class D3D12AccelerationStructure : public AccelerationStructure {
 public:
  D3D12AccelerationStructure(D3D12Core *core, std::unique_ptr<d3d12::AccelerationStructure> &&acceleration_structure);

  int UpdateInstances(const std::vector<RayTracingInstance> &instances) override;

  d3d12::AccelerationStructure *Handle() const {
    return acceleration_structure_.get();
  }

 private:
  D3D12Core *core_;
  std::unique_ptr<d3d12::AccelerationStructure> acceleration_structure_;
};

}  // namespace grassland::graphics::backend
