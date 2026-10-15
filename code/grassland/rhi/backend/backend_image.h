#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_image.h"

namespace grassland::rhi::backend {

class Image : public Object<rhi::IImage> {
 public:
  explicit Image(ImageDesc desc);
  const ImageDesc &GetDesc() const override;
  virtual bool IsValid() const = 0;
  virtual const void *DeviceIdentity() const = 0;

 private:
  ImageDesc desc_;
};

}  // namespace grassland::rhi::backend
