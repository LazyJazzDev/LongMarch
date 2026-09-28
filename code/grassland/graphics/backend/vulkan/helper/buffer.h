#pragma once
#include "grassland/graphics/backend/vulkan/helper/native_types.h"

namespace grassland::graphics::backend::vulkan {
class Buffer {
 public:
  Buffer(const VulkanCore *device, VkDeviceSize size, VkBuffer buffer, VmaAllocation allocation);

  ~Buffer();

  const VulkanCore *Device() const {
    return device_;
  }

  VkBuffer Handle() const {
    return buffer_;
  }

  VmaAllocation Allocation() const {
    return allocation_;
  }

  VkDeviceSize Size() const {
    return size_;
  }

  void *Map() const;
  void Unmap() const;

  VkDeviceAddress GetDeviceAddress() const;

 private:
  const VulkanCore *device_{};
  VkDeviceSize size_{};
  VkBuffer buffer_{};
  VmaAllocation allocation_{};
};

void CopyBuffer(VkCommandBuffer command_buffer,
                Buffer *src_buffer,
                Buffer *dst_buffer,
                VkDeviceSize size,
                VkDeviceSize src_offset = 0,
                VkDeviceSize dst_offset = 0);

void UploadBuffer(VkQueue queue, VkCommandPool command_pool, Buffer *buffer, const void *data, VkDeviceSize size);

void DownloadBuffer(VkQueue queue, VkCommandPool command_pool, Buffer *buffer, void *data, VkDeviceSize size);

class BufferObject {
 public:
  virtual Buffer *GetBuffer(uint32_t frame_index) const = 0;
};

#if defined(LONGMARCH_CUDA_RUNTIME)
VkExternalMemoryHandleTypeFlagBits GetDefaultExternalMemoryHandleType();

void CreateExternalBuffer(VkDevice device,
                          std::function<uint32_t(uint32_t, VkMemoryPropertyFlags)> find_memory_type,
                          VkDeviceSize size,
                          VkBufferUsageFlags usage,
                          VkMemoryPropertyFlags properties,
                          VkBuffer &buffer,
                          VkDeviceMemory &bufferMemory);
#endif

}  // namespace grassland::graphics::backend::vulkan
