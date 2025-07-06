#pragma once
#include "grassland/d3d12/d3d12util.h"

namespace grassland::d3d12 {

class RayTracingPipeline {
 public:
  RayTracingPipeline(const ComPtr<ID3D12StateObject> &state_object);
  ~RayTracingPipeline() = default;

  ID3D12StateObject *Handle() const {
    return state_object_.Get();
  }

 private:
  ComPtr<ID3D12StateObject> state_object_;
};

}  // namespace grassland::d3d12
