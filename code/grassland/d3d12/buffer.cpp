#include "grassland/d3d12/buffer.h"

namespace grassland::d3d12 {

Buffer::Buffer(const ComPtr<ID3D12Resource> &buffer) : buffer_(buffer) {
}

void *Buffer::Map() const {
  void *data;
  buffer_->Map(0, nullptr, &data);
  return data;
}

void Buffer::Unmap() const {
  buffer_->Unmap(0, nullptr);
}

}  // namespace grassland::d3d12
