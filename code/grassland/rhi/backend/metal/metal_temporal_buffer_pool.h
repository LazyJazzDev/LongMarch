#pragma once
#include "grassland/rhi/backend/backend_temporal_buffer_pool.h"
#include "grassland/rhi/backend/metal/metal_allocated_buffer.h"
#include "grassland/rhi/backend/metal/metal_util.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::metal::IMetalTemporalBufferLease,
                        grassland::rhi::backend::metal::IMetalBuffer,
                        "d8febe53-823a-4731-847d-f2f1766778c8")

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
