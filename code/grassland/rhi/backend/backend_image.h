#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_image.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::IBackendImage)

namespace grassland::rhi::backend {
class IBackendImage : public IObject {
 public:
  virtual bool SupportsQueue(const IBackendQueue *queue) const = 0;
  virtual bool SupportsStorage() const = 0;
  virtual const void *DeviceIdentity() const = 0;
};

}  // namespace grassland::rhi::backend
