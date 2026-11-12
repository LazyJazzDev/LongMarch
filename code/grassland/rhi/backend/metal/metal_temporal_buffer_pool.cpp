#include "grassland/rhi/backend/metal/metal_temporal_buffer_pool.h"

#include "grassland/rhi/backend/metal/metal_device.h"

namespace grassland::rhi::backend::metal {

IMetalTemporalBufferLease *IMetalTemporalBufferLease::Create(MetalTemporalBufferPool *pool,
                                                             IMetalAllocatedBuffer *buffer,
                                                             size_t level) {
  return new MetalTemporalBufferLeaseImpl(pool, buffer, level);
}

void IMetalTemporalBufferLease::ReturnBuffer(MetalTemporalBufferPool *pool,
                                             IMetalAllocatedBuffer *buffer,
                                             size_t level) {
  pool->ReleaseBuffer(buffer, level);
}

MetalTemporalBufferLeaseImpl::MetalTemporalBufferLeaseImpl(MetalTemporalBufferPool *pool,
                                                           IMetalAllocatedBuffer *buffer,
                                                           size_t level)
    : pool_(pool),
      buffer_(buffer),
      level_(level) {
}

MetalTemporalBufferLeaseImpl::~MetalTemporalBufferLeaseImpl() {
  ReturnBuffer(pool_, buffer_, level_);
}

BufferDesc MetalTemporalBufferLeaseImpl::GetDesc() const {
  return {MetalTemporalBufferPool::CapacityForLevel(level_)};
}

MTL::Buffer *MetalTemporalBufferLeaseImpl::NativeBuffer() const {
  return buffer_->NativeBuffer();
}

bool MetalTemporalBufferLeaseImpl::Write(const void *data, uint64_t size) {
  return buffer_->Write(data, size);
}

bool MetalTemporalBufferLeaseImpl::Read(void *data, uint64_t size, uint64_t offset) {
  return buffer_->Read(data, size, offset);
}
}  // namespace grassland::rhi::backend::metal
