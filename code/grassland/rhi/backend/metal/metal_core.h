#pragma once
#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/backend_core.h"
#include "grassland/rhi/backend/metal/metal_util.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::metal::IMetalCore)

namespace grassland::rhi::backend::metal {

class IMetalCore : public IObject {
 public:
  static rhi::ICore *CreateMetalCore(const CoreDesc &desc);
  virtual bool DebugEnabled() const = 0;
};

}  // namespace grassland::rhi::backend::metal
