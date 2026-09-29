#include "grassland/graphics/backend/webgpu/webgpu_buffer.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "grassland/graphics/backend/webgpu/webgpu_core.h"

namespace grassland::graphics::backend {

WebGPUBuffer::WebGPUBuffer(WebGPUCore *core, size_t size, BufferType type) : core_(core), type_(type) {
  Resize(size);
}

void WebGPUBuffer::Resize(size_t size) {
  if (buffer_ && size == size_)
    return;
  wgpu::BufferDescriptor descriptor{};
  descriptor.size = std::max<uint64_t>(16, (uint64_t(size) + 15) & ~uint64_t(15));
  descriptor.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::Storage | wgpu::BufferUsage::Vertex |
                     wgpu::BufferUsage::Index | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc;
  auto replacement = core_->Device().CreateBuffer(&descriptor);
  if (buffer_) {
    // Preserve the old contents in queue order, like the other backends' Resize.
    auto encoder = core_->Device().CreateCommandEncoder();
    encoder.CopyBufferToBuffer(buffer_, 0, replacement, 0, (std::min<uint64_t>(size, size_) + 3) & ~uint64_t(3));
    auto commands = encoder.Finish();
    core_->Queue().Submit(1, &commands);
  }
  buffer_ = std::move(replacement);
  size_ = size;
}

void WebGPUBuffer::UploadData(const void *data, size_t size, size_t offset) {
  if (offset > size_ || size > size_ - offset)
    throw std::out_of_range("WebGPU buffer upload");
  if (!size)
    return;
  if (offset % 4)
    throw std::invalid_argument("WebGPU buffer uploads start at a multiple of 4 bytes");
  // Queue writes must be a multiple of 4 bytes; the allocation always has room.
  if (size % 4) {
    std::vector<uint8_t> padded((size + 3) & ~size_t(3), 0);
    std::memcpy(padded.data(), data, size);
    core_->Queue().WriteBuffer(buffer_, offset, padded.data(), padded.size());
  } else {
    core_->Queue().WriteBuffer(buffer_, offset, data, size);
  }
}

void WebGPUBuffer::DownloadData(void *, size_t, size_t) {
  throw std::runtime_error("WebGPU buffer readback is asynchronous and unavailable here");
}

}  // namespace grassland::graphics::backend
