#include "grassland/d3d12/command_list.h"

namespace grassland::d3d12 {

CommandList::CommandList(const ComPtr<ID3D12GraphicsCommandList> &command_list)
    : command_list_(command_list) {
}

}  // namespace grassland::d3d12
