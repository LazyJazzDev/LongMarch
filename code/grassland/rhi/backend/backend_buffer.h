#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_buffer.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::IBackendBuffer,
                        grassland::rhi::IObject,
                        "a86fae4e-f66b-43e7-89b4-3d5c431bf0d3")

namespace grassland::rhi::backend {
class IBackendBuffer : public IObject {
 public:
  virtual IBackendDevice *Owner() const = 0;
};
}  // namespace grassland::rhi::backend
