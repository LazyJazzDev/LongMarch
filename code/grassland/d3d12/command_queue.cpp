#include "grassland/d3d12/command_queue.h"

namespace grassland::d3d12 {

CommandQueue::CommandQueue(const ComPtr<ID3D12CommandQueue> &command_queue)
    : command_queue_(command_queue) {
}

}  // namespace grassland::d3d12
