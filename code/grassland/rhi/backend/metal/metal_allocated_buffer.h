#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend::metal {
class IMetalAllocatedBuffer;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::metal::IMetalAllocatedBuffer)
#include "grassland/rhi/backend/metal/metal_buffer.h"

namespace grassland::rhi::backend::metal {
class IMetalAllocatedBuffer : public IMetalBuffer {
 public:
  static IMetalAllocatedBuffer *Create(MetalDevice *device,
                                       const BufferDesc &desc,
                                       bool upload = false,
                                       bool download = false);
  virtual MTL::IBackendBuffer *NativeBuffer() const = 0;
  virtual bool Read(void *data, uint64_t size, uint64_t offset = 0) = 0;
  virtual bool Write(const void *data, uint64_t size) = 0;
};
}  // namespace grassland::rhi::backend::metal
