#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_image.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::IBackendImage,
                        grassland::rhi::IObject,
                        "1c62d59a-7c6e-4442-8855-d6b1d1b4804c")

namespace grassland::rhi::backend {
class IBackendImage : public IObject {
 public:
  virtual bool SupportsQueue(const IBackendQueue *queue) const = 0;
  virtual bool SupportsStorage() const = 0;
  virtual const void *DeviceIdentity() const = 0;
};

}  // namespace grassland::rhi::backend
