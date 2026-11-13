#include "grassland/rhi/backend/vulkan/vulkan_allocated_buffer.h"

#include <vk_mem_alloc.h>

#include "grassland/rhi/backend/vulkan/vulkan_device.h"

namespace grassland::rhi::backend::vulkan {

IVulkanAllocatedBuffer *IVulkanAllocatedBuffer::Create(IVulkanDevice *device,
                                                       const BufferDesc &desc,
                                                       bool upload,
                                                       bool download) {
  return VulkanAllocatedBufferImpl::Create(device, desc, upload, download);
}

IVulkanDevice *VulkanAllocatedBufferImpl::Owner() const {
  return device_;
}

VulkanAllocatedBufferImpl::VulkanAllocatedBufferImpl(IVulkanDevice *device, const BufferDesc &desc)
    : device_(device),
      desc_(desc) {
}

VulkanAllocatedBufferImpl::~VulkanAllocatedBufferImpl() {
  if (buffer_)
    vmaDestroyBuffer(device_->Allocator(), buffer_, allocation_);
}

VkBuffer VulkanAllocatedBufferImpl::NativeBuffer() const {
  return buffer_;
}

IVulkanAllocatedBuffer *VulkanAllocatedBufferImpl::Create(IVulkanDevice *device,
                                                          const BufferDesc &desc,
                                                          bool upload,
                                                          bool download) {
  if (!desc.size)
    return nullptr;
  auto object = std::unique_ptr<VulkanAllocatedBufferImpl>(new VulkanAllocatedBufferImpl(device, desc));
  VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  info.size = object->desc_.size;
  info.usage = download ? VK_BUFFER_USAGE_TRANSFER_DST_BIT
               : upload ? VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                        : VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                              VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  if (device->GetPhysicalDevice().ray_tracing_pipeline_feature || device->GetPhysicalDevice().ray_query_feature) {
    info.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                  VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    if (device->GetPhysicalDevice().ray_tracing_pipeline_feature)
      info.usage |= VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR;
  }
  device->ConfigureSharing(info);
  VmaAllocationCreateInfo memory{};
  memory.usage = (upload || download) ? VMA_MEMORY_USAGE_AUTO_PREFER_HOST : VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  if (upload)
    memory.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
  if (download)
    memory.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
  VmaAllocationInfo allocation{};
  if (!Check(vmaCreateBuffer(object->device_->Allocator(), &info, &memory, &object->buffer_, &object->allocation_,
                             &allocation),
             "Allocate buffer"))
    return nullptr;
  object->mapped_ = allocation.pMappedData;
  return object.release();
}

bool VulkanAllocatedBufferImpl::Write(const void *data, uint64_t size) {
  if (!mapped_ || size > desc_.size)
    return false;
  std::memcpy(mapped_, data, static_cast<size_t>(size));
  return Check(vmaFlushAllocation(device_->Allocator(), allocation_, 0, size), "Flush upload buffer");
}

bool VulkanAllocatedBufferImpl::Read(void *data, uint64_t size, uint64_t offset) {
  if (!mapped_ || !data || offset > desc_.size || size > desc_.size - offset)
    return false;
  if (!Check(vmaInvalidateAllocation(device_->Allocator(), allocation_, offset, size), "Invalidate readback buffer"))
    return false;
  std::memcpy(data, static_cast<const uint8_t *>(mapped_) + offset, static_cast<size_t>(size));
  return true;
}
}  // namespace grassland::rhi::backend::vulkan
