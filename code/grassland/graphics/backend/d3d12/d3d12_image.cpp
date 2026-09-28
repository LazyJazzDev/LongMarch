#include "grassland/graphics/backend/d3d12/d3d12_image.h"

namespace grassland::graphics::backend {

namespace {
void *MapBuffer(ID3D12Resource *buffer) {
  void *data = nullptr;
  d3d12::ThrowIfFailed(buffer->Map(0, nullptr, &data), "Failed to map image staging buffer");
  return data;
}
}  // namespace

D3D12Image::D3D12Image(D3D12Core *core, int width, int height, ImageFormat format) : core_(core), format_(format) {
  image_ = CreateNativeImage(core_->Device()->Handle(), width, height, ImageFormatToDXGIFormat(format));
}

Extent2D D3D12Image::Extent() const {
  Extent2D extent;
  extent.width = image_->GetDesc().Width;
  extent.height = image_->GetDesc().Height;
  return extent;
}

ImageFormat D3D12Image::Format() const {
  return format_;
}

void D3D12Image::UploadData(const void *data) const {
  auto pixel_size = PixelSize(format_);
  const UINT64 upload_buffer_size = GetRequiredIntermediateSize(image_.Get(), 0, 1);
  Microsoft::WRL::ComPtr<ID3D12Resource> upload_buffer;
  upload_buffer = CreateNativeBuffer(core_->Device()->Handle(), upload_buffer_size, D3D12_HEAP_TYPE_UPLOAD);
  D3D12_SUBRESOURCE_DATA subresource_data{};
  subresource_data.pData = data;
  subresource_data.RowPitch = image_->GetDesc().Width * pixel_size;
  subresource_data.SlicePitch = subresource_data.RowPitch * image_->GetDesc().Height;

  core_->SingleTimeCommand([&](ID3D12GraphicsCommandList *command_list) {
    CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
        image_.Get(), D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATE_COPY_DEST);
    command_list->ResourceBarrier(1, &barrier);
    UpdateSubresources(command_list, image_.Get(), upload_buffer.Get(), 0, 0, 1, &subresource_data);
    barrier = CD3DX12_RESOURCE_BARRIER::Transition(image_.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                                   D3D12_RESOURCE_STATE_GENERIC_READ);
    command_list->ResourceBarrier(1, &barrier);
  });
}

void D3D12Image::DownloadData(void *data) const {
  auto pixel_size = PixelSize(format_);
  const UINT64 download_buffer_size = GetRequiredIntermediateSize(image_.Get(), 0, 1);
  Microsoft::WRL::ComPtr<ID3D12Resource> download_buffer;
  download_buffer = CreateNativeBuffer(core_->Device()->Handle(), download_buffer_size, D3D12_HEAP_TYPE_READBACK);
  D3D12_SUBRESOURCE_DATA subresource_data{};
  subresource_data.pData = data;
  subresource_data.RowPitch = image_->GetDesc().Width * pixel_size;
  subresource_data.SlicePitch = subresource_data.RowPitch * image_->GetDesc().Height;

  D3D12_TEXTURE_COPY_LOCATION src_location{};
  src_location.pResource = image_.Get();
  src_location.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  src_location.SubresourceIndex = 0;

  D3D12_TEXTURE_COPY_LOCATION dst_location{};
  dst_location.pResource = download_buffer.Get();
  dst_location.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;

  D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout{};
  auto desc = image_.Get()->GetDesc();
  core_->Device()->Handle()->GetCopyableFootprints(&desc, 0, 1, 0, &layout, nullptr, nullptr, nullptr);
  dst_location.PlacedFootprint = layout;

  core_->SingleTimeCommand([&](ID3D12GraphicsCommandList *command_list) {
    CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
        image_.Get(), D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATE_COPY_SOURCE);
    command_list->ResourceBarrier(1, &barrier);

    command_list->CopyTextureRegion(&dst_location, 0, 0, 0, &src_location, nullptr);

    barrier = CD3DX12_RESOURCE_BARRIER::Transition(image_.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                                                   D3D12_RESOURCE_STATE_GENERIC_READ);
    command_list->ResourceBarrier(1, &barrier);
  });

  uint8_t *mapped_data = static_cast<uint8_t *>(MapBuffer(download_buffer.Get()));
  for (UINT row = 0; row < image_->GetDesc().Height; row++) {
    memcpy(static_cast<uint8_t *>(data) + row * subresource_data.RowPitch,
           mapped_data + layout.Offset + row * layout.Footprint.RowPitch, subresource_data.RowPitch);
  }
  download_buffer->Unmap(0, nullptr);
}

void D3D12Image::UploadData(const void *data, const Offset2D &offset, const Extent2D &extent) const {
  auto pixel_size = PixelSize(format_);

  // Create a staging image that matches the region size
  Microsoft::WRL::ComPtr<ID3D12Resource> staging_image;
  staging_image =
      CreateNativeImage(core_->Device()->Handle(), extent.width, extent.height, ImageFormatToDXGIFormat(format_));

  // Create upload buffer sized for the staging image
  const UINT64 upload_buffer_size = GetRequiredIntermediateSize(staging_image.Get(), 0, 1);
  Microsoft::WRL::ComPtr<ID3D12Resource> upload_buffer;
  upload_buffer = CreateNativeBuffer(core_->Device()->Handle(), upload_buffer_size, D3D12_HEAP_TYPE_UPLOAD);

  // Calculate source data layout
  D3D12_SUBRESOURCE_DATA subresource_data{};
  subresource_data.pData = data;
  subresource_data.RowPitch = extent.width * pixel_size;
  subresource_data.SlicePitch = subresource_data.RowPitch * extent.height;

  core_->SingleTimeCommand([&](ID3D12GraphicsCommandList *command_list) {
    // Transition staging image to copy destination for upload
    CD3DX12_RESOURCE_BARRIER staging_barrier = CD3DX12_RESOURCE_BARRIER::Transition(
        staging_image.Get(), D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATE_COPY_DEST);
    command_list->ResourceBarrier(1, &staging_barrier);

    // Upload data to staging image
    UpdateSubresources(command_list, staging_image.Get(), upload_buffer.Get(), 0, 0, 1, &subresource_data);

    // Transition staging image to copy source
    staging_barrier = CD3DX12_RESOURCE_BARRIER::Transition(staging_image.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                                           D3D12_RESOURCE_STATE_COPY_SOURCE);
    command_list->ResourceBarrier(1, &staging_barrier);

    // Transition main image to copy destination
    CD3DX12_RESOURCE_BARRIER main_barrier = CD3DX12_RESOURCE_BARRIER::Transition(
        image_.Get(), D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATE_COPY_DEST);
    command_list->ResourceBarrier(1, &main_barrier);

    // Copy from staging image to main image at specified offset
    D3D12_TEXTURE_COPY_LOCATION src_location{};
    src_location.pResource = staging_image.Get();
    src_location.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src_location.SubresourceIndex = 0;

    D3D12_TEXTURE_COPY_LOCATION dst_location{};
    dst_location.pResource = image_.Get();
    dst_location.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst_location.SubresourceIndex = 0;

    // Copy entire staging image to the specified region
    D3D12_BOX src_box{};
    src_box.left = 0;
    src_box.top = 0;
    src_box.front = 0;
    src_box.right = extent.width;
    src_box.bottom = extent.height;
    src_box.back = 1;

    command_list->CopyTextureRegion(&dst_location, offset.x, offset.y, 0, &src_location, &src_box);

    // Transition main image back to generic read
    main_barrier = CD3DX12_RESOURCE_BARRIER::Transition(image_.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                                        D3D12_RESOURCE_STATE_GENERIC_READ);
    command_list->ResourceBarrier(1, &main_barrier);
  });
}

void D3D12Image::DownloadData(void *data, const Offset2D &offset, const Extent2D &extent) const {
  auto pixel_size = PixelSize(format_);

  // Create a staging image that matches the region size
  Microsoft::WRL::ComPtr<ID3D12Resource> staging_image;
  staging_image =
      CreateNativeImage(core_->Device()->Handle(), extent.width, extent.height, ImageFormatToDXGIFormat(format_));

  // Create download buffer sized for the staging image
  const UINT64 download_buffer_size = GetRequiredIntermediateSize(staging_image.Get(), 0, 1);
  Microsoft::WRL::ComPtr<ID3D12Resource> download_buffer;
  download_buffer = CreateNativeBuffer(core_->Device()->Handle(), download_buffer_size, D3D12_HEAP_TYPE_READBACK);

  // Calculate destination data layout
  D3D12_SUBRESOURCE_DATA subresource_data{};
  subresource_data.pData = data;
  subresource_data.RowPitch = extent.width * pixel_size;
  subresource_data.SlicePitch = subresource_data.RowPitch * extent.height;

  D3D12_TEXTURE_COPY_LOCATION src_location{};
  src_location.pResource = image_.Get();
  src_location.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  src_location.SubresourceIndex = 0;

  D3D12_TEXTURE_COPY_LOCATION dst_location{};
  dst_location.pResource = staging_image.Get();
  dst_location.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  dst_location.SubresourceIndex = 0;

  // Copy from staging image to download buffer
  D3D12_TEXTURE_COPY_LOCATION staging_src_location{};
  staging_src_location.pResource = staging_image.Get();
  staging_src_location.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  staging_src_location.SubresourceIndex = 0;

  D3D12_TEXTURE_COPY_LOCATION buffer_dst_location{};
  buffer_dst_location.pResource = download_buffer.Get();
  buffer_dst_location.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;

  D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout{};
  auto staging_desc = staging_image.Get()->GetDesc();
  core_->Device()->Handle()->GetCopyableFootprints(&staging_desc, 0, 1, 0, &layout, nullptr, nullptr, nullptr);
  buffer_dst_location.PlacedFootprint = layout;

  core_->SingleTimeCommand([&](ID3D12GraphicsCommandList *command_list) {
    // Transition staging image to copy destination
    CD3DX12_RESOURCE_BARRIER staging_barrier = CD3DX12_RESOURCE_BARRIER::Transition(
        staging_image.Get(), D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATE_COPY_DEST);
    command_list->ResourceBarrier(1, &staging_barrier);

    // Transition main image to copy source
    CD3DX12_RESOURCE_BARRIER main_barrier = CD3DX12_RESOURCE_BARRIER::Transition(
        image_.Get(), D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATE_COPY_SOURCE);
    command_list->ResourceBarrier(1, &main_barrier);

    // Copy from main image to staging image at specified offset
    D3D12_BOX src_box{};
    src_box.left = offset.x;
    src_box.top = offset.y;
    src_box.front = 0;
    src_box.right = offset.x + extent.width;
    src_box.bottom = offset.y + extent.height;
    src_box.back = 1;

    command_list->CopyTextureRegion(&dst_location, 0, 0, 0, &src_location, &src_box);

    // Transition staging image to copy source
    staging_barrier = CD3DX12_RESOURCE_BARRIER::Transition(staging_image.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                                           D3D12_RESOURCE_STATE_COPY_SOURCE);
    command_list->ResourceBarrier(1, &staging_barrier);

    // Copy from staging image to download buffer
    D3D12_BOX staging_box{};
    staging_box.left = 0;
    staging_box.top = 0;
    staging_box.front = 0;
    staging_box.right = extent.width;
    staging_box.bottom = extent.height;
    staging_box.back = 1;

    command_list->CopyTextureRegion(&buffer_dst_location, 0, 0, 0, &staging_src_location, &staging_box);

    // Transition main image back to generic read after all operations are finished
    main_barrier = CD3DX12_RESOURCE_BARRIER::Transition(image_.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                                                        D3D12_RESOURCE_STATE_GENERIC_READ);
    command_list->ResourceBarrier(1, &main_barrier);
  });

  // Copy data from download buffer to user buffer
  uint8_t *mapped_data = static_cast<uint8_t *>(MapBuffer(download_buffer.Get()));
  for (UINT row = 0; row < extent.height; row++) {
    memcpy(static_cast<uint8_t *>(data) + row * subresource_data.RowPitch,
           mapped_data + layout.Offset + row * layout.Footprint.RowPitch, subresource_data.RowPitch);
  }
  download_buffer->Unmap(0, nullptr);
}

ID3D12Resource *D3D12Image::Image() const {
  return image_.Get();
}

}  // namespace grassland::graphics::backend
