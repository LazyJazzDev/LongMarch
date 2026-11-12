#include "grassland/rhi/backend/d3d12/d3d12_temporal_buffer_pool.h"

#include "grassland/rhi/backend/d3d12/d3d12_device.h"

namespace grassland::rhi::backend::d3d12 {

ID3D12TemporalBufferLease *ID3D12TemporalBufferLease::Create(D3D12TemporalBufferPool *pool,
                                                             ID3D12AllocatedBuffer *buffer,
                                                             size_t level) {
  return new D3D12TemporalBufferLeaseImpl(pool, buffer, level);
}

void ID3D12TemporalBufferLease::ReturnBuffer(D3D12TemporalBufferPool *pool,
                                             ID3D12AllocatedBuffer *buffer,
                                             size_t level) {
  pool->ReleaseBuffer(buffer, level);
}

D3D12TemporalBufferLeaseImpl::D3D12TemporalBufferLeaseImpl(D3D12TemporalBufferPool *pool,
                                                           ID3D12AllocatedBuffer *buffer,
                                                           size_t level)
    : pool_(pool),
      buffer_(buffer),
      level_(level) {
}

D3D12TemporalBufferLeaseImpl::~D3D12TemporalBufferLeaseImpl() {
  ReturnBuffer(pool_, buffer_, level_);
}

BufferDesc D3D12TemporalBufferLeaseImpl::GetDesc() const {
  return {D3D12TemporalBufferPool::CapacityForLevel(level_)};
}

ID3D12Resource *D3D12TemporalBufferLeaseImpl::NativeBuffer() const {
  return buffer_->NativeBuffer();
}

D3D12_RESOURCE_STATES D3D12TemporalBufferLeaseImpl::DefaultState() const {
  return buffer_->DefaultState();
}

bool D3D12TemporalBufferLeaseImpl::Write(const void *data, uint64_t size, uint64_t offset) {
  return buffer_->Write(data, size, offset);
}

bool D3D12TemporalBufferLeaseImpl::Read(void *data, uint64_t size, uint64_t offset) {
  return buffer_->Read(data, size, offset);
}
}  // namespace grassland::rhi::backend::d3d12
