#pragma once
#include "grassland/graphics/backend/d3d12/d3d12_core.h"
#include "grassland/graphics/backend/d3d12/d3d12_util.h"

namespace CD::graphics::backend {

class D3D12Image : public Image {
 public:
  D3D12Image(D3D12Core *core, int width, int height, ImageFormat format);
  Extent2D Extent() const override;
  ImageFormat Format() const override;
  void UploadData(const void *data) const override;
  void DownloadData(void *data) const override;

  d3d12::Image *Image() const;

 private:
  D3D12Core *core_;
  std::unique_ptr<d3d12::Image> image_;
  ImageFormat format_;
};

}  // namespace CD::graphics::backend
