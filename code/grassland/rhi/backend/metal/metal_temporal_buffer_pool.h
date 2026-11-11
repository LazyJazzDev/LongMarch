#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend::metal {
class IMetalTemporalBufferLease;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::metal::IMetalTemporalBufferLease)
#include "grassland/rhi/backend/backend_temporal_buffer_pool.h"
#include "grassland/rhi/backend/metal/metal_allocated_buffer.h"

namespace grassland::rhi::backend::metal {
class IMetalTemporalBufferLease;
using MetalTemporalBufferPool = TemporalBufferPool<MetalDevice, IMetalAllocatedBuffer, IMetalTemporalBufferLease>;

class IMetalTemporalBufferLease : public IMetalBuffer {
 public:
  static IMetalTemporalBufferLease *Create(MetalTemporalBufferPool *pool, IMetalAllocatedBuffer *buffer, size_t level);
  virtual BufferDesc GetDesc() const = 0;
  virtual MTL::IBackendBuffer *NativeBuffer() const = 0;
  virtual bool Read(void *data, uint64_t size, uint64_t offset = 0) = 0;
  virtual bool Write(const void *data, uint64_t size) = 0;

 protected:
  static void ReturnBuffer(MetalTemporalBufferPool *pool, IMetalAllocatedBuffer *buffer, size_t level);
};

}  // namespace grassland::rhi::backend::metal
