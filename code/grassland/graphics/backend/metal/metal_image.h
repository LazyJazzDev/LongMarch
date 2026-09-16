#pragma once
#include "grassland/graphics/backend/metal/metal_util.h"

namespace grassland::graphics::backend {

class MetalImage : public Image {
 public:
  MetalImage(MetalCore *core, int width, int height, ImageFormat format);
  Extent2D Extent() const override {
    return extent_;
  }
  ImageFormat Format() const override {
    return format_;
  }
  void UploadData(const void *data) const override {
    UploadData(data, {0, 0}, extent_);
  }
  void DownloadData(void *data) const override {
    DownloadData(data, {0, 0}, extent_);
  }
  void UploadData(const void *data, const Offset2D &offset, const Extent2D &extent) const override;
  void DownloadData(void *data, const Offset2D &offset, const Extent2D &extent) const override;
  MTL::Texture *Handle() const {
    return texture_.get();
  }

 private:
  MetalCore *core_;
  Extent2D extent_;
  ImageFormat format_;
  NS::SharedPtr<MTL::Texture> texture_;
};

}  // namespace grassland::graphics::backend
