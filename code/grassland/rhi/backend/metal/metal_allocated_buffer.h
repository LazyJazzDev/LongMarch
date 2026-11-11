#pragma once
#include "grassland/rhi/backend/metal/metal_buffer.h"
#include "grassland/rhi/backend/metal/metal_util.h"

LONGMARCH_RHI_UUID(grassland::rhi::backend::metal::IMetalAllocatedBuffer, "19d20694-8b38-47a4-a88f-56d5828c0e9d")

namespace grassland::rhi::backend::metal {
class IMetalAllocatedBuffer : public IMetalBuffer {
 public:
  static IMetalAllocatedBuffer *Create(IMetalDevice *device,
                                       const BufferDesc &desc,
                                       bool upload = false,
                                       bool download = false);

  virtual bool Read(void *data, uint64_t size, uint64_t offset = 0) = 0;
  virtual bool Write(const void *data, uint64_t size) = 0;
};
}  // namespace grassland::rhi::backend::metal
