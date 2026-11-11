#pragma once
#include "grassland/rhi/rhi_object.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::IImage, grassland::rhi::IObject, "150542bb-1f11-4f68-b1d8-e2db200cc22d")

namespace grassland::rhi {
class IImage : public IObject {
 public:
  virtual const ImageDesc &GetDesc() const = 0;
  // Swapchain frames are single-use; they must not be held across resize.
  // Detecting reuse after Present is a Debug-only diagnostic.
  virtual bool IsValid() const = 0;

 protected:
  ~IImage() override;
};

}  // namespace grassland::rhi
