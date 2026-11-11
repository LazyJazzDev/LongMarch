#pragma once
#include "grassland/rhi/backend/backend_temporal_buffer_pool.h"
#include "grassland/rhi/backend/metal/metal_allocated_buffer.h"
#include "grassland/rhi/backend/metal/metal_util.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::metal::IMetalTemporalBufferLease)

namespace grassland::rhi::backend::metal {
using MetalTemporalBufferPool = TemporalBufferPool<IMetalDevice, IMetalAllocatedBuffer, IMetalTemporalBufferLease>;

class IMetalTemporalBufferLease : public IMetalBuffer {
 public:
  static IMetalTemporalBufferLease *Create(MetalTemporalBufferPool *pool, IMetalAllocatedBuffer *buffer, size_t level);

  virtual bool Read(void *data, uint64_t size, uint64_t offset = 0) = 0;
  virtual bool Write(const void *data, uint64_t size) = 0;

 protected:
  static void ReturnBuffer(MetalTemporalBufferPool *pool, IMetalAllocatedBuffer *buffer, size_t level);
};

}  // namespace grassland::rhi::backend::metal
