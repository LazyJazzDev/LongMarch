#pragma once
#include "grassland/graphics/backend/vulkan/vulkan_core.h"
#include "grassland/graphics/backend/vulkan/vulkan_util.h"

namespace grassland::graphics::backend {

class VulkanBuffer : public Buffer {
 public:
  virtual VkBuffer Buffer() const = 0;
};

class VulkanStaticBuffer : public VulkanBuffer {
 public:
  VulkanStaticBuffer(VulkanCore *core, size_t size);
  ~VulkanStaticBuffer() override;

  size_t Size() const override {
    return buffer_->Size();
  }

  BufferType Type() const override {
    return BUFFER_TYPE_STATIC;
  }

  void Resize(size_t new_size) override;

  void UploadData(const void *data, size_t size, size_t offset) override;

  void DownloadData(void *data, size_t size, size_t offset) override;

  VkBuffer Buffer() const override;

 private:
  VulkanCore *core_;
  std::unique_ptr<vulkan::Buffer> buffer_;
};
}  // namespace grassland::graphics::backend
