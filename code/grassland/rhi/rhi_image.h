#pragma once
#include "grassland/rhi/rhi_object.h"

namespace grassland::rhi {
class IImage : public IObject {
 public:
  virtual const ImageDesc &GetDesc() const = 0;
  // Swapchain frames are single-use; they must not be held across resize.
  virtual bool IsValid() const = 0;

 protected:
  ~IImage() override;
};

}  // namespace grassland::rhi
