#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_interface_id.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::IBackendImage)
#include "grassland/rhi/rhi_image.h"

namespace grassland::rhi::backend {
struct ImageState;

class IBackendImage : public rhi::IImage {
 protected:
  void InitializeImage(ImageDesc desc, IBackendQueue *queue = nullptr);

 public:
  bool SupportsQueue(const IBackendQueue *queue) const;
  const ImageDesc &GetDesc() const override = 0;
  const ImageDesc &CommonGetDesc() const;
  virtual bool SupportsStorage() const = 0;
  bool CommonSupportsStorage() const;
  virtual const void *DeviceIdentity() const = 0;

 protected:
  virtual ImageState &StateImage() = 0;
  virtual const ImageState &StateImage() const = 0;
};

}  // namespace grassland::rhi::backend
