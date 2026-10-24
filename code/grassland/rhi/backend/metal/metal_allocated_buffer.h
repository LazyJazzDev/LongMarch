#pragma once
#include "grassland/rhi/backend/metal/metal_buffer.h"

namespace grassland::rhi::backend::metal {
class MetalAllocatedBuffer final : public MetalBuffer {
 public:
  static MetalAllocatedBuffer *Create(MetalDevice *device,
                                      const BufferDesc &desc,
                                      bool upload = false,
                                      bool download = false);
  MTL::Buffer *NativeBuffer() const override;
  bool Read(void *data, uint64_t size, uint64_t offset = 0);
  bool Write(const void *data, uint64_t size);

 private:
  MetalAllocatedBuffer(MetalDevice *device, const BufferDesc &desc);
  NS::SharedPtr<MTL::Buffer> buffer_;
};
}  // namespace grassland::rhi::backend::metal
