#pragma once
#include "grassland/rhi/backend/backend_temporal_buffer_pool.h"
#include "grassland/rhi/backend/d3d12/d3d12_allocated_buffer.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12TemporalBufferLease;
using D3D12TemporalBufferPool = TemporalBufferPool<D3D12Device, D3D12AllocatedBuffer, D3D12TemporalBufferLease>;

class D3D12TemporalBufferLease final : public D3D12Buffer {
 public:
  D3D12TemporalBufferLease(D3D12TemporalBufferPool *pool, D3D12AllocatedBuffer *buffer, size_t level);
  ~D3D12TemporalBufferLease() override;
  BufferDesc GetDesc() const override;
  ID3D12Resource *NativeBuffer() const override;
  D3D12_RESOURCE_STATES DefaultState() const override;
  bool Read(void *data, uint64_t size, uint64_t offset = 0);
  bool Write(const void *data, uint64_t size, uint64_t offset = 0);

 private:
  D3D12TemporalBufferPool *pool_;
  D3D12AllocatedBuffer *buffer_;
  size_t level_;
};

}  // namespace grassland::rhi::backend::d3d12
