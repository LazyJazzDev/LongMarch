#pragma once
#include "grassland/rhi/backend/backend_buffer.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12Device;

// Resource access only. Derived classes define allocation ownership (standalone or future pool leases).
class D3D12Buffer : public Buffer {
 public:
  virtual ID3D12Resource *NativeBuffer() const = 0;
  virtual D3D12_RESOURCE_STATES DefaultState() const = 0;

 protected:
  D3D12Buffer(D3D12Device *device, const BufferDesc &desc);
};
}  // namespace grassland::rhi::backend::d3d12
