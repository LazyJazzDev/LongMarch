#pragma once
#include "grassland/graphics/backend/webgpu/webgpu_util.h"

namespace grassland::graphics::backend {

class WebGPUBuffer : public Buffer {
 public:
  WebGPUBuffer(WebGPUCore *core, size_t size, BufferType type);

  BufferType Type() const override {
    return type_;
  }

  size_t Size() const override {
    return size_;
  }

  void Resize(size_t size) override;
  void UploadData(const void *data, size_t size, size_t offset = 0) override;
  // WebGPU maps buffers asynchronously; a browser thread cannot wait for it.
  void DownloadData(void *data, size_t size, size_t offset = 0) override;

  const wgpu::Buffer &Handle() const {
    return buffer_;
  }

  // The allocation is rounded up so uniform bindings cover WGSL's 16-byte struct sizes.
  uint64_t AllocatedSize() const {
    return buffer_.GetSize();
  }

 private:
  WebGPUCore *core_;
  size_t size_ = 0;
  BufferType type_;
  wgpu::Buffer buffer_;
};

}  // namespace grassland::graphics::backend
