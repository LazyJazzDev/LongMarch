#include "grassland/graphics/backend/d3d12/helper/pipeline_state.h"

namespace grassland::graphics::backend::d3d12 {

PipelineState::PipelineState(const ComPtr<ID3D12PipelineState> &pipeline_state) : pipeline_state_(pipeline_state) {
}

}  // namespace grassland::graphics::backend::d3d12
