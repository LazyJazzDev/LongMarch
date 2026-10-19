#include "grassland/rhi/backend/d3d12/d3d12_buffer.h"

#include "grassland/rhi/backend/d3d12/d3d12_device.h"

namespace grassland::rhi::backend::d3d12 {
D3D12Buffer::D3D12Buffer(D3D12Device *device, const BufferDesc &desc) : Buffer(device, desc) {
}
}  // namespace grassland::rhi::backend::d3d12
