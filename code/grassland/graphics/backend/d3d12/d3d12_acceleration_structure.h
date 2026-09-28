#pragma once
#include "grassland/graphics/backend/d3d12/d3d12_util.h"

namespace grassland::graphics::backend {

class D3D12AccelerationStructure : public AccelerationStructure {
 public:
  D3D12AccelerationStructure(D3D12Core *core, ComPtr<ID3D12Resource> acceleration_structure, int instance_count);

  int UpdateInstances(const std::vector<RayTracingInstance> &instances) override;

  ID3D12Resource *Handle() const {
    return acceleration_structure_.Get();
  }

 private:
  D3D12Core *core_;
  ComPtr<ID3D12Resource> acceleration_structure_;
  int instance_count_{};
};

}  // namespace grassland::graphics::backend
