#include "grassland/vulkan/buffer.h"

#include "grassland/vulkan/single_time_command.h"

#ifdef _WIN64
#include <VersionHelpers.h>
#include <dxgi1_2.h>
#endif

namespace CD::vulkan {
Buffer::Buffer(const class Device *device, VkDeviceSize size, VkBuffer buffer, VmaAllocation allocation)
    : device_(device), size_(size), buffer_(buffer), allocation_(allocation) {
}

Buffer::~Buffer() {
  vmaDestroyBuffer(device_->Allocator(), buffer_, allocation_);
}

VkDeviceAddress Buffer::GetDeviceAddress() const {
  VkBufferDeviceAddressInfo buffer_device_address_info{};
  buffer_device_address_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
  buffer_device_address_info.buffer = buffer_;
  return vkGetBufferDeviceAddress(device_->Handle(), &buffer_device_address_info);
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
  vkCmdCopyBuffer(command_buffer, src_buffer->Handle(), dst_buffer->Handle(), 1, &copy_region);
}

void UploadBuffer(Queue *queue, CommandPool *command_pool, Buffer *buffer, const void *data, VkDeviceSize size) {
  std::unique_ptr<Buffer> staging_buffer;
  buffer->Device()->CreateBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY, &staging_buffer);
  void *staging_data = staging_buffer->Map();
  std::memcpy(staging_data, data, size);
  staging_buffer->Unmap();

  SingleTimeCommand(queue, command_pool,
                    [&](VkCommandBuffer cmd_buffer) { CopyBuffer(cmd_buffer, staging_buffer.get(), buffer, size); });
}

void DownloadBuffer(Queue *queue, CommandPool *command_pool, Buffer *buffer, void *data, VkDeviceSize size) {
  std::unique_ptr<Buffer> staging_buffer;
  buffer->Device()->CreateBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_ONLY, &staging_buffer);
  SingleTimeCommand(queue, command_pool,
                    [&](VkCommandBuffer cmd_buffer) { CopyBuffer(cmd_buffer, buffer, staging_buffer.get(), size); });
  void *staging_data = staging_buffer->Map();
  std::memcpy(data, staging_data, size);
  staging_buffer->Unmap();
}

#if defined(LONGMARCH_CUDA_RUNTIME)
VkExternalMemoryHandleTypeFlagBits GetDefaultExternalMemoryHandleType() {
#ifdef _WIN64
  return IsWindows8Point1OrGreater() ? VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
                                     : VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT;
#else
  return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif /* _WIN64 */
}

void CreateExternalBuffer(VkDevice device,
                          std::function<uint32_t(uint32_t, VkMemoryPropertyFlags)> find_memory_type,
                          VkDeviceSize size,
                          VkBufferUsageFlags usage,
                          VkMemoryPropertyFlags properties,
                          VkBuffer &buffer,
                          VkDeviceMemory &bufferMemory) {
  auto handle_type = GetDefaultExternalMemoryHandleType();
  VkBufferCreateInfo buffer_info = {};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = size;
  buffer_info.usage = usage;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VkExternalMemoryBufferCreateInfo external_memory_buffer_info = {};
  external_memory_buffer_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
  external_memory_buffer_info.handleTypes = handle_type;
  buffer_info.pNext = &external_memory_buffer_info;

  if (vkCreateBuffer(device, &buffer_info, nullptr, &buffer) != VK_SUCCESS) {
    throw std::runtime_error("failed to create buffer!");
  }

  VkMemoryRequirements memory_requirements;
  vkGetBufferMemoryRequirements(device, buffer, &memory_requirements);

#ifdef _WIN64
  WindowsSecurityAttributes winSecurityAttributes;

  VkExportMemoryWin32HandleInfoKHR vulkan_export_memory_win32_handle_info_khr = {};
  vulkan_export_memory_win32_handle_info_khr.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
  vulkan_export_memory_win32_handle_info_khr.pNext = NULL;
  vulkan_export_memory_win32_handle_info_khr.pAttributes = &winSecurityAttributes;
  vulkan_export_memory_win32_handle_info_khr.dwAccess = DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;
  vulkan_export_memory_win32_handle_info_khr.name = (LPCWSTR)NULL;
#endif /* _WIN64 */
  VkExportMemoryAllocateInfoKHR export_memory_allocate_info = {};
  export_memory_allocate_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
#ifdef _WIN64
  export_memory_allocate_info.pNext = handle_type & VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR
                                          ? &vulkan_export_memory_win32_handle_info_khr
                                          : NULL;
  export_memory_allocate_info.handleTypes = handle_type;
#else
  export_memory_allocate_info.pNext = NULL;
  export_memory_allocate_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif /* _WIN64 */
  VkMemoryAllocateInfo alloc_info = {};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.pNext = &export_memory_allocate_info;
  alloc_info.allocationSize = memory_requirements.size;
  alloc_info.memoryTypeIndex = find_memory_type(memory_requirements.memoryTypeBits, properties);

  VkMemoryAllocateFlagsInfo memory_allocate_flags_info = {};
  if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
    memory_allocate_flags_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    memory_allocate_flags_info.pNext = alloc_info.pNext;
    memory_allocate_flags_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    alloc_info.pNext = &memory_allocate_flags_info;
  }

  if (vkAllocateMemory(device, &alloc_info, nullptr, &bufferMemory) != VK_SUCCESS) {
    throw std::runtime_error("failed to allocate external buffer memory!");
  }

  vkBindBufferMemory(device, buffer, bufferMemory, 0);
}
#endif

}  // namespace CD::vulkan
