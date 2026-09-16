#pragma once
#include "grassland/graphics/backend/metal/metal_util.h"

namespace grassland::graphics::backend {

class MetalBuffer : public Buffer {
 public:
  MetalBuffer(MetalCore *core, size_t size, BufferType type);
  BufferType Type() const override {
    return type_;
  }
  size_t Size() const override {
    return size_;
  }
  void Resize(size_t size) override;
  void UploadData(const void *data, size_t size, size_t offset = 0) override;
  void DownloadData(void *data, size_t size, size_t offset = 0) override;
  MTL::Buffer *Handle() const {
    return buffer_.get();
  }

 private:
  MetalCore *core_;
  size_t size_ = 0;
  BufferType type_;
  NS::SharedPtr<MTL::Buffer> buffer_;
};

}  // namespace grassland::graphics::backend
