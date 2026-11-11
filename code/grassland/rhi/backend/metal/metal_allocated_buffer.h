#pragma once
#include "grassland/rhi/backend/metal/metal_buffer.h"
#include "grassland/rhi/backend/metal/metal_util.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::metal::IMetalAllocatedBuffer)

namespace grassland::rhi::backend::metal {
class IMetalAllocatedBuffer : public IMetalBuffer {
 public:
  static IMetalAllocatedBuffer *Create(IMetalDevice *device,
                                       const BufferDesc &desc,
                                       bool upload = false,
                                       bool download = false);
  virtual MTL::Buffer *NativeBuffer() const = 0;
  virtual bool Read(void *data, uint64_t size, uint64_t offset = 0) = 0;
  virtual bool Write(const void *data, uint64_t size) = 0;
};
}  // namespace grassland::rhi::backend::metal
