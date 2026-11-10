#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_image.h"

namespace grassland::rhi::backend {
class Queue;

class Image : public Object<rhi::IImage> {
 public:
  explicit Image(ImageDesc desc, Queue *queue = nullptr);
  bool SupportsQueue(const Queue *queue) const;
  const ImageDesc &GetDesc() const override;
  virtual bool SupportsStorage() const;
  virtual const void *DeviceIdentity() const = 0;

 private:
  ImageDesc desc_;
  Queue *queue_;
};

}  // namespace grassland::rhi::backend
