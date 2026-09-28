#include "grassland/graphics/backend/d3d12/helper/command_list.h"

namespace grassland::graphics::backend::d3d12 {

CommandList::CommandList(const ComPtr<ID3D12GraphicsCommandList> &command_list) : command_list_(command_list) {
}

}  // namespace grassland::graphics::backend::d3d12
