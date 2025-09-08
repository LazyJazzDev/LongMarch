#include "grassland/d3d12/pipeline_state.h"

namespace CD::d3d12 {

PipelineState::PipelineState(const ComPtr<ID3D12PipelineState> &pipeline_state) : pipeline_state_(pipeline_state) {
}

}  // namespace CD::d3d12
