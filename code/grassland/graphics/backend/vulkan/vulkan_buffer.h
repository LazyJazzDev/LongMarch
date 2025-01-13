#pragma once
#include "grassland/graphics/backend/vulkan/vulkan_core.h"
#include "grassland/graphics/backend/vulkan/vulkan_util.h"

namespace grassland::graphics::backend {

class VulkanBuffer : public Buffer {
 public:
  virtual VkBuffer Buffer() const = 0;
  virtual ~VulkanBuffer() = default;

  virtual vulkan::Buffer *InstantBuffer() const = 0;
};

class VulkanStaticBuffer : public VulkanBuffer {
 public:
  VulkanStaticBuffer(VulkanCore *core, size_t size);
  ~VulkanStaticBuffer() override;

  size_t Size() const override;

  BufferType Type() const override;

  void Resize(size_t new_size) override;

  void UploadData(const void *data, size_t size, size_t offset) override;

  void DownloadData(void *data, size_t size, size_t offset) override;

  VkBuffer Buffer() const override;

  vulkan::Buffer *InstantBuffer() const override;

 private:
  VulkanCore *core_;
  std::unique_ptr<vulkan::Buffer> buffer_;
};

class VulkanDynamicBuffer : public VulkanBuffer {
 public:
  VulkanDynamicBuffer(VulkanCore *core, size_t size);
  ~VulkanDynamicBuffer() override;

  size_t Size() const override;

  BufferType Type() const override;

  void Resize(size_t new_size) override;

  void UploadData(const void *data, size_t size, size_t offset) override;

  void DownloadData(void *data, size_t size, size_t offset) override;

  VkBuffer Buffer() const override;

  vulkan::Buffer *InstantBuffer() const override;

  void TransferData(VkCommandBuffer cmd_buffer);

 private:
  VulkanCore *core_;
  std::vector<std::unique_ptr<vulkan::Buffer>> buffers_;
  std::unique_ptr<vulkan::Buffer> staging_buffer_;
};

}  // namespace grassland::graphics::backend
