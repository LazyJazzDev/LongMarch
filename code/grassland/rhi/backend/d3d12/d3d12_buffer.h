#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend::d3d12 {
class ID3D12Buffer;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::d3d12::ID3D12Buffer)
#include "grassland/rhi/backend/backend_buffer.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {
class ID3D12Device;

// Resource access only. Derived classes define allocation ownership (standalone or future pool leases).
class ID3D12Buffer : public IBackendBuffer {
 public:
  virtual ID3D12Resource *NativeBuffer() const = 0;
  virtual D3D12_RESOURCE_STATES DefaultState() const = 0;
};
}  // namespace grassland::rhi::backend::d3d12
