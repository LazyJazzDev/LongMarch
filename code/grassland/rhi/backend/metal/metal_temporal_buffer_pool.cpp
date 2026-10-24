#include "grassland/rhi/backend/metal/metal_temporal_buffer_pool.h"

#include "grassland/rhi/backend/metal/metal_device.h"

namespace grassland::rhi::backend::metal {
MetalTemporalBufferLease::MetalTemporalBufferLease(MetalTemporalBufferPool *pool,
                                                   MetalAllocatedBuffer *buffer,
                                                   size_t level)
    : MetalBuffer(static_cast<MetalDevice *>(buffer->Owner()), buffer->GetDesc()),
      pool_(pool),
      buffer_(buffer),
      level_(level) {
}

MetalTemporalBufferLease::~MetalTemporalBufferLease() {
  pool_->ReleaseBuffer(buffer_, level_);
}

BufferDesc MetalTemporalBufferLease::GetDesc() const {
  return buffer_->GetDesc();
}

MTL::Buffer *MetalTemporalBufferLease::NativeBuffer() const {
  return buffer_->NativeBuffer();
}

bool MetalTemporalBufferLease::Write(const void *data, uint64_t size) {
  return buffer_->Write(data, size);
}

bool MetalTemporalBufferLease::Read(void *data, uint64_t size, uint64_t offset) {
  return buffer_->Read(data, size, offset);
}
}  // namespace grassland::rhi::backend::metal
