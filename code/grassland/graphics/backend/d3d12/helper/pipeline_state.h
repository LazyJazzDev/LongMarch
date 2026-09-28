#pragma once
#include "grassland/graphics/backend/d3d12/helper/device.h"

namespace grassland::graphics::backend::d3d12 {

class PipelineState {
 public:
  PipelineState(const ComPtr<ID3D12PipelineState> &pipeline_state);

  ID3D12PipelineState *Handle() const {
    return pipeline_state_.Get();
  }

 private:
  ComPtr<ID3D12PipelineState> pipeline_state_;
};

}  // namespace grassland::graphics::backend::d3d12
