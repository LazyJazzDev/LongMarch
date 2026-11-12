#pragma once
#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/metal/metal_util.h"
#include "grassland/rhi/rhi_buffer.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::metal::IMetalBuffer,
                        grassland::rhi::IObject,
                        "203e6dfb-bb02-4155-900a-b13f43af47bc")

namespace grassland::rhi::backend::metal {
class IMetalBuffer : public IObject {
 public:
  virtual IMetalDevice *Owner() const = 0;
  virtual MTL::Buffer *NativeBuffer() const = 0;
};
}  // namespace grassland::rhi::backend::metal
