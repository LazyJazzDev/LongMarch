#include "grassland/rhi/backend/d3d12/d3d12_queue.h"

#include "grassland/rhi/backend/d3d12/d3d12_device.h"

namespace grassland::rhi::backend::d3d12 {

D3D12Queue::D3D12Queue(D3D12Device *device, QueueType type) : Queue(device, type) {
}

}  // namespace grassland::rhi::backend::d3d12
