#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend::metal {
class MetalTemporalBufferLease;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::metal::MetalTemporalBufferLease)
#include "grassland/rhi/backend/backend_temporal_buffer_pool.h"
#include "grassland/rhi/backend/metal/metal_allocated_buffer.h"

namespace grassland::rhi::backend::metal {
class MetalTemporalBufferLease;
using MetalTemporalBufferPool = TemporalBufferPool<MetalDevice, MetalAllocatedBuffer, MetalTemporalBufferLease>;

class MetalTemporalBufferLease final : public MetalBuffer {
 public:
  LONGMARCH_RHI_QUERY(MetalTemporalBufferLease, MetalBuffer)

  MetalTemporalBufferLease(MetalTemporalBufferPool *pool, MetalAllocatedBuffer *buffer, size_t level);
  ~MetalTemporalBufferLease() override;
  BufferDesc GetDesc() const override;
  MTL::Buffer *NativeBuffer() const override;
  bool Read(void *data, uint64_t size, uint64_t offset = 0);
  bool Write(const void *data, uint64_t size);

 private:
  MetalTemporalBufferPool *pool_;
  MetalAllocatedBuffer *buffer_;
  size_t level_;
};

}  // namespace grassland::rhi::backend::metal
