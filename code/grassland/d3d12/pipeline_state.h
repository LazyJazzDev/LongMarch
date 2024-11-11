#pragma once
#include "grassland/d3d12/device.h"

namespace grassland::d3d12 {

class PipelineState {
 public:
  PipelineState(const ComPtr<ID3D12PipelineState> &pipeline_state);

  ID3D12PipelineState *Handle() const {
    return pipeline_state_.Get();
  }

 private:
  ComPtr<ID3D12PipelineState> pipeline_state_;
};

}  // namespace grassland::d3d12
