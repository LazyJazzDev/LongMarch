#include "grassland/graphics/backend/d3d12/d3d12_image.h"

namespace grassland::graphics::backend {

D3D12Image::D3D12Image(D3D12Core *core,
                       int width,
                       int height,
                       ImageFormat format)
    : core_(core), format_(format) {
  core_->Device()->CreateImage(width, height, ImageFormatToDXGIFormat(format),
                               &image_);
}

Extent2D D3D12Image::Extent() const {
  Extent2D extent;
  extent.width = image_->Width();
  extent.height = image_->Height();
  return extent;
}

ImageFormat D3D12Image::Format() const {
  return format_;
}

void D3D12Image::UploadData(const void *data) const {
  auto pixel_size = PixelSize(format_);
  const UINT64 upload_buffer_size =
      GetRequiredIntermediateSize(image_->Handle(), 0, 1);
  std::unique_ptr<d3d12::Buffer> upload_buffer;
  core_->Device()->CreateBuffer(upload_buffer_size, D3D12_HEAP_TYPE_UPLOAD,
                                &upload_buffer);
  D3D12_SUBRESOURCE_DATA subresource_data{};
  subresource_data.pData = data;
  subresource_data.RowPitch = image_->Width() * pixel_size;
  subresource_data.SlicePitch = subresource_data.RowPitch * image_->Height();

  core_->SingleTimeCommand([&](ID3D12GraphicsCommandList *command_list) {
    CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
        image_->Handle(), D3D12_RESOURCE_STATE_GENERIC_READ,
        D3D12_RESOURCE_STATE_COPY_DEST);
    command_list->ResourceBarrier(1, &barrier);
    UpdateSubresources(command_list, image_->Handle(), upload_buffer->Handle(),
                       0, 0, 1, &subresource_data);
    barrier = CD3DX12_RESOURCE_BARRIER::Transition(
        image_->Handle(), D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_GENERIC_READ);
    command_list->ResourceBarrier(1, &barrier);
  });
}

d3d12::Image *D3D12Image::Image() const {
  return image_.get();
}

}  // namespace grassland::graphics::backend
