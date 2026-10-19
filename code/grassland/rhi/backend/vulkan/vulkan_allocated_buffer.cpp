#include "grassland/rhi/backend/vulkan/vulkan_allocated_buffer.h"

#include <vk_mem_alloc.h>

#include "grassland/rhi/backend/vulkan/vulkan_device.h"

namespace grassland::rhi::backend::vulkan {
VulkanAllocatedBuffer::VulkanAllocatedBuffer(VulkanDevice *device, const BufferDesc &desc)
    : VulkanBuffer(device, desc) {
}

VulkanAllocatedBuffer::~VulkanAllocatedBuffer() {
  if (buffer_)
    vmaDestroyBuffer(static_cast<VulkanDevice *>(device_)->Allocator(), buffer_, allocation_);
}

VkBuffer VulkanAllocatedBuffer::NativeBuffer() const {
  return buffer_;
}

bool VulkanAllocatedBuffer::Initialize(bool upload) {
  VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  info.size = desc_.size;
  info.usage = upload ? VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                      : VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  VmaAllocationCreateInfo memory{};
  memory.usage = upload ? VMA_MEMORY_USAGE_AUTO_PREFER_HOST : VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  if (upload)
    memory.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
  VmaAllocationInfo allocation{};
  if (!Check(vmaCreateBuffer(static_cast<VulkanDevice *>(device_)->Allocator(), &info, &memory, &buffer_, &allocation_,
                             &allocation),
             "Allocate buffer"))
    return false;
  mapped_ = allocation.pMappedData;
  return true;
}

bool VulkanAllocatedBuffer::Write(const void *data, uint64_t size) {
  if (!mapped_ || size > desc_.size)
    return false;
  std::memcpy(mapped_, data, static_cast<size_t>(size));
  return Check(vmaFlushAllocation(static_cast<VulkanDevice *>(device_)->Allocator(), allocation_, 0, size),
               "Flush upload buffer");
}
}  // namespace grassland::rhi::backend::vulkan
