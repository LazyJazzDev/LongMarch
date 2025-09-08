#pragma once
#include "grassland/graphics/backend/vulkan/vulkan_core.h"
#include "grassland/graphics/backend/vulkan/vulkan_util.h"

namespace CD::graphics::backend {

struct VulkanBufferRange {
  VulkanBuffer *buffer;
  VkDeviceSize offset;
  VkDeviceSize size;
  VulkanBufferRange() = default;
  VulkanBufferRange(const BufferRange &range);
};

class VulkanBuffer : virtual public Buffer {
 public:
  virtual ~VulkanBuffer() = default;

  virtual VkBuffer Buffer() const = 0;
  virtual VkDeviceAddress DeviceAddress() const = 0;
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

  VkDeviceAddress DeviceAddress() const override;

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

  VkDeviceAddress DeviceAddress() const override;

  void TransferData(VkCommandBuffer cmd_buffer);

 private:
  VulkanCore *core_;
  std::vector<std::unique_ptr<vulkan::Buffer>> buffers_;
  std::unique_ptr<vulkan::Buffer> staging_buffer_;
};

#if defined(LONGMARCH_CUDA_RUNTIME)
class VulkanCUDABuffer : public VulkanBuffer, public CUDABuffer {
 public:
  VulkanCUDABuffer(VulkanCore *core, size_t size);
  ~VulkanCUDABuffer() override;

  void Reset();

  size_t Size() const override;

  BufferType Type() const override;

  void Resize(size_t new_size) override;

  void UploadData(const void *data, size_t size, size_t offset) override;

  void DownloadData(void *data, size_t size, size_t offset) override;

  VkBuffer Buffer() const override;

  VkDeviceAddress DeviceAddress() const override;

  void GetCUDAMemoryPointer(void **ptr) override;

 private:
  VulkanCore *core_;
  VkBuffer buffer_;
  VkDeviceMemory memory_;
  VkDeviceAddress address_;
  VkDeviceSize size_;
  cudaExternalMemory_t cuda_memory_;
};
#endif

}  // namespace CD::graphics::backend
