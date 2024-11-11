#include "grassland/vulkan/buffer.h"

#include "grassland/vulkan/single_time_command.h"

namespace grassland::vulkan {
Buffer::Buffer(const class Device *device,
               VkDeviceSize size,
               VkBuffer buffer,
               VmaAllocation allocation)
    : device_(device), size_(size), buffer_(buffer), allocation_(allocation) {
}

Buffer::~Buffer() {
  vmaDestroyBuffer(device_->Allocator(), buffer_, allocation_);
}

VkDeviceAddress Buffer::GetDeviceAddress() const {
  VkBufferDeviceAddressInfo buffer_device_address_info{};
  buffer_device_address_info.sType =
      VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
  buffer_device_address_info.buffer = buffer_;
  return vkGetBufferDeviceAddress(device_->Handle(),
                                  &buffer_device_address_info);
}

void CopyBuffer(VkCommandBuffer command_buffer,
                Buffer *src_buffer,
                Buffer *dst_buffer,
                VkDeviceSize size,
                VkDeviceSize src_offset,
                VkDeviceSize dst_offset) {
  VkBufferCopy copy_region{};
  copy_region.size = size;
  copy_region.srcOffset = src_offset;
  copy_region.dstOffset = dst_offset;
  vkCmdCopyBuffer(command_buffer, src_buffer->Handle(), dst_buffer->Handle(), 1,
                  &copy_region);
}

void UploadBuffer(Queue *queue,
                  CommandPool *command_pool,
                  Buffer *buffer,
                  const void *data,
                  VkDeviceSize size) {
  std::unique_ptr<Buffer> staging_buffer;
  buffer->Device()->CreateBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                 VMA_MEMORY_USAGE_CPU_ONLY, &staging_buffer);
  void *staging_data = staging_buffer->Map();
  std::memcpy(staging_data, data, size);
  staging_buffer->Unmap();

  SingleTimeCommand(queue, command_pool, [&](VkCommandBuffer cmd_buffer) {
    CopyBuffer(cmd_buffer, staging_buffer.get(), buffer, size);
  });
}

void DownloadBuffer(Queue *queue,
                    CommandPool *command_pool,
                    Buffer *buffer,
                    void *data,
                    VkDeviceSize size) {
  std::unique_ptr<Buffer> staging_buffer;
  buffer->Device()->CreateBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                 VMA_MEMORY_USAGE_CPU_ONLY, &staging_buffer);
  SingleTimeCommand(queue, command_pool, [&](VkCommandBuffer cmd_buffer) {
    CopyBuffer(cmd_buffer, buffer, staging_buffer.get(), size);
  });
  void *staging_data = staging_buffer->Map();
  std::memcpy(data, staging_data, size);
  staging_buffer->Unmap();
}
}  // namespace grassland::vulkan
