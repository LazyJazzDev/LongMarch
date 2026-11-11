#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend::d3d12 {
class ID3D12TemporalBufferLease;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::d3d12::ID3D12TemporalBufferLease)
#include "grassland/rhi/backend/backend_temporal_buffer_pool.h"
#include "grassland/rhi/backend/d3d12/d3d12_allocated_buffer.h"

namespace grassland::rhi::backend::d3d12 {
class ID3D12TemporalBufferLease;
using D3D12TemporalBufferPool = TemporalBufferPool<D3D12Device, ID3D12AllocatedBuffer, ID3D12TemporalBufferLease>;

class ID3D12TemporalBufferLease : public ID3D12Buffer {
 public:
  static ID3D12TemporalBufferLease *Create(D3D12TemporalBufferPool *pool, ID3D12AllocatedBuffer *buffer, size_t level);
  virtual BufferDesc GetDesc() const = 0;
  virtual ID3D12Resource *NativeBuffer() const = 0;
  virtual D3D12_RESOURCE_STATES DefaultState() const = 0;
  virtual bool Read(void *data, uint64_t size, uint64_t offset = 0) = 0;
  virtual bool Write(const void *data, uint64_t size, uint64_t offset = 0) = 0;

 protected:
  static void ReturnBuffer(D3D12TemporalBufferPool *pool, ID3D12AllocatedBuffer *buffer, size_t level);
};

}  // namespace grassland::rhi::backend::d3d12
