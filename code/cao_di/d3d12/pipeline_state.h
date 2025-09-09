#pragma once
#include "cao_di/d3d12/device.h"

namespace CD::d3d12 {

class PipelineState {
 public:
  PipelineState(const ComPtr<ID3D12PipelineState> &pipeline_state);

  ID3D12PipelineState *Handle() const {
    return pipeline_state_.Get();
  }

 private:
  ComPtr<ID3D12PipelineState> pipeline_state_;
};

}  // namespace CD::d3d12
