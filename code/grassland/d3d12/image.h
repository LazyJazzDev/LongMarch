#pragma once
#include "grassland/d3d12/device.h"

namespace CD::d3d12 {

class Image {
 public:
  Image(const ComPtr<ID3D12Resource> &image);

  ID3D12Resource *Handle() const {
    return image_.Get();
  }

  size_t Width() const {
    return image_->GetDesc().Width;
  }

  size_t Height() const {
    return image_->GetDesc().Height;
  }

  DXGI_FORMAT Format() const {
    return image_->GetDesc().Format;
  }

  size_t PixelSize() const {
    return SizeByFormat(Format());
  }

  size_t UploadBufferSize() const {
    return GetRequiredIntermediateSize(image_.Get(), 0, 1);
  }

 private:
  ComPtr<ID3D12Resource> image_;
};

void UploadImageData(ID3D12GraphicsCommandList *command_list, Buffer *buffer, Image *image);

void DownloadImageData(ID3D12GraphicsCommandList *command_list, Image *image, Buffer *buffer);

}  // namespace CD::d3d12
