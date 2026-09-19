#include "grassland/graphics/backend/metal/metal_buffer.h"

#include <cstring>
#include <stdexcept>

#include "grassland/graphics/backend/metal/metal_core.h"

namespace grassland::graphics::backend {

MetalBuffer::MetalBuffer(MetalCore *core, size_t size, BufferType type) : core_(core), type_(type) {
  Resize(size);
}

void MetalBuffer::Resize(size_t size) {
  if (buffer_ && size == size_)
    return;
  core_->WaitGPU();
  MetalPool pool;
  auto replacement =
      NS::TransferPtr(core_->Device()->newBuffer(std::max(size_t(4), size), MTL::ResourceStorageModeShared));
  MetalCheck(replacement.get(), nullptr, "newBuffer");
  if (buffer_)
    std::memcpy(replacement->contents(), buffer_->contents(), std::min(size, size_));
  buffer_ = std::move(replacement);
  size_ = size;
}

void MetalBuffer::UploadData(const void *data, size_t size, size_t offset) {
  if (offset > size_ || size > size_ - offset)
    throw std::out_of_range("Metal buffer upload");
  core_->WaitGPU();
  if (size)
    std::memcpy(static_cast<char *>(buffer_->contents()) + offset, data, size);
}

void MetalBuffer::DownloadData(void *data, size_t size, size_t offset) {
  if (offset > size_ || size > size_ - offset)
    throw std::out_of_range("Metal buffer download");
  core_->WaitGPU();
  if (size)
    std::memcpy(data, static_cast<char *>(buffer_->contents()) + offset, size);
}

}  // namespace grassland::graphics::backend
