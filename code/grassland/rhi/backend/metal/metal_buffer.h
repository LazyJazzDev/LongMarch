#pragma once
#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/backend_buffer.h"
#include "grassland/rhi/backend/metal/metal_util.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::metal::IMetalBuffer)

namespace grassland::rhi::backend::metal {
class IMetalBuffer : public IBackendBuffer {
 public:
  virtual MTL::Buffer *NativeBuffer() const = 0;
};
}  // namespace grassland::rhi::backend::metal
