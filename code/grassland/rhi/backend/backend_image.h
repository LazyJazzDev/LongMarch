#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend {
class Image;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::Image)
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_image.h"

namespace grassland::rhi::backend {
class Queue;

struct ImageState;

class Image : public rhi::IImage {
 protected:
  void InitializeImage(ImageDesc desc, Queue *queue = nullptr);

 public:
  bool SupportsQueue(const Queue *queue) const;
  const ImageDesc &GetDesc() const override;
  virtual bool SupportsStorage() const;
  virtual const void *DeviceIdentity() const = 0;

 protected:
  virtual ImageState &StateImage() = 0;
  virtual const ImageState &StateImage() const = 0;
};

}  // namespace grassland::rhi::backend
