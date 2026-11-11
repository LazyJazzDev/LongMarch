#include "grassland/rhi/backend/d3d12/d3d12_buffer.h"

#include "grassland/rhi/backend/d3d12/d3d12_device.h"

namespace grassland::rhi::backend::d3d12 {
D3D12Buffer::D3D12Buffer(D3D12Device *device, const BufferDesc &desc) : Buffer(desc), device_(device) {
}

Device *D3D12Buffer::Owner() const {
  return device_;
}

}  // namespace grassland::rhi::backend::d3d12

LONGMARCH_RHI_UUID(grassland::rhi::backend::d3d12::D3D12Buffer, "29cbc4ee-c77c-48a4-8c31-de9ee75c704d")
