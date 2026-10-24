#include "grassland/rhi/backend/d3d12/d3d12_temporal_buffer_pool.h"

#include "grassland/rhi/backend/d3d12/d3d12_device.h"

namespace grassland::rhi::backend::d3d12 {
D3D12TemporalBufferLease::D3D12TemporalBufferLease(D3D12TemporalBufferPool *pool,
                                                   D3D12AllocatedBuffer *buffer,
                                                   size_t level)
    : D3D12Buffer(static_cast<D3D12Device *>(buffer->Owner()), buffer->GetDesc()),
      pool_(pool),
      buffer_(buffer),
      level_(level) {
}

D3D12TemporalBufferLease::~D3D12TemporalBufferLease() {
  pool_->ReleaseBuffer(buffer_, level_);
}

BufferDesc D3D12TemporalBufferLease::GetDesc() const {
  return buffer_->GetDesc();
}

ID3D12Resource *D3D12TemporalBufferLease::NativeBuffer() const {
  return buffer_->NativeBuffer();
}

D3D12_RESOURCE_STATES D3D12TemporalBufferLease::DefaultState() const {
  return buffer_->DefaultState();
}

bool D3D12TemporalBufferLease::Write(const void *data, uint64_t size, uint64_t offset) {
  return buffer_->Write(data, size, offset);
}

bool D3D12TemporalBufferLease::Read(void *data, uint64_t size, uint64_t offset) {
  return buffer_->Read(data, size, offset);
}
}  // namespace grassland::rhi::backend::d3d12
