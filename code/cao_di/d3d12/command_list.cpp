#include "cao_di/d3d12/command_list.h"

namespace CD::d3d12 {

CommandList::CommandList(const ComPtr<ID3D12GraphicsCommandList> &command_list) : command_list_(command_list) {
}

}  // namespace CD::d3d12
