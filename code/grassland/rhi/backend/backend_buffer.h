#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_interface_id.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::IBackendBuffer)
#include "grassland/rhi/rhi_buffer.h"

namespace grassland::rhi::backend {
class IBackendBuffer : public IBuffer {
 public:
  virtual IBackendDevice *Owner() const = 0;
};
}  // namespace grassland::rhi::backend
