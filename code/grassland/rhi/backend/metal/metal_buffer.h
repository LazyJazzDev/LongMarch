#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_interface_id.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::metal::IMetalBuffer)
#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/backend_buffer.h"

namespace grassland::rhi::backend::metal {
class IMetalBuffer : public IBackendBuffer {
 public:
  virtual MTL::Buffer *NativeBuffer() const = 0;
};
}  // namespace grassland::rhi::backend::metal
