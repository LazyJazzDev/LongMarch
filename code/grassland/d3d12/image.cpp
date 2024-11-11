#include "grassland/d3d12/image.h"

#include "grassland/d3d12/buffer.h"

namespace grassland::d3d12 {

Image::Image(const ComPtr<ID3D12Resource> &image) : image_(image) {
}

void UploadImageData(ID3D12GraphicsCommandList *command_list,
                     Buffer *buffer,
                     Image *image) {
  D3D12_SUBRESOURCE_DATA subresource_data = {};
  subresource_data.pData =
      image->Handle()->GetDesc().Layout == D3D12_TEXTURE_LAYOUT_ROW_MAJOR
          ? buffer->Map()
          : nullptr;
}

}  // namespace grassland::d3d12
