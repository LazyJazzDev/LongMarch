#pragma once
#include "cao_di/vulkan/device.h"

namespace CD::vulkan {
class Buffer {
 public:
  Buffer(const class Device *device, VkDeviceSize size, VkBuffer buffer, VmaAllocation allocation);

  ~Buffer();

  const class Device *Device() const {
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

  void *Map() const {
    void *data;
    vmaMapMemory(device_->Allocator(), allocation_, &data);
    return data;
  }

  void Unmap() const {
    vmaUnmapMemory(device_->Allocator(), allocation_);
  }

  VkDeviceAddress GetDeviceAddress() const;

 private:
  const class Device *device_{};
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

void UploadBuffer(Queue *queue, CommandPool *command_pool, Buffer *buffer, const void *data, VkDeviceSize size);

void DownloadBuffer(Queue *queue, CommandPool *command_pool, Buffer *buffer, void *data, VkDeviceSize size);

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

}  // namespace CD::vulkan
