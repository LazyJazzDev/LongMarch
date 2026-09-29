#pragma once
#include "grassland/graphics/backend/webgpu/webgpu_util.h"

namespace grassland::graphics::backend {

class WebGPUImage : public Image {
 public:
  WebGPUImage(WebGPUCore *core, int width, int height, ImageFormat format);

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
  // WebGPU maps buffers asynchronously; a browser thread cannot wait for it.
  void DownloadData(void *data, const Offset2D &offset, const Extent2D &extent) const override;

  const wgpu::Texture &Handle() const {
    return texture_;
  }

  const wgpu::TextureView &View() const {
    return view_;
  }

 private:
  WebGPUCore *core_;
  Extent2D extent_;
  ImageFormat format_;
  wgpu::Texture texture_;
  wgpu::TextureView view_;
};

}  // namespace grassland::graphics::backend
