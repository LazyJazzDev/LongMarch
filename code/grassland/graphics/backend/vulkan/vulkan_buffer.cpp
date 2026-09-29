#include "grassland/graphics/backend/vulkan/vulkan_buffer.h"

#include "grassland/graphics/frame_profile.h"

namespace grassland::graphics::backend {

VulkanBufferRange::VulkanBufferRange(const BufferRange &range)
    : buffer(dynamic_cast<VulkanBuffer *>(range.buffer)),
      offset(range.offset),
      size(range.size) {
}

namespace {
VkBufferUsageFlags StaticUsage(VulkanCore *core) {
  VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                             VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                             VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  if (core->DeviceRayTracingSupport() || core->DeviceRayQuerySupport())
    usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
             VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
  return usage;
}

VkBufferUsageFlags DynamicStagingUsage(VulkanCore *core) {
  VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  if (core->DeviceRayTracingSupport() || core->DeviceRayQuerySupport())
    usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
             VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
  return usage;
}
}  // namespace

VulkanStaticBuffer::VulkanStaticBuffer(VulkanCore *core, size_t size) : core_(core), size_(size) {
  vulkan::ThrowIfFailed(core_->CreateBuffer(size, StaticUsage(core), VMA_MEMORY_USAGE_GPU_ONLY, &buffer_, &allocation_),
                        "Failed to create Vulkan static buffer");
}

VulkanStaticBuffer::~VulkanStaticBuffer() {
  vmaDestroyBuffer(core_->Allocator(), buffer_, allocation_);
}

size_t VulkanStaticBuffer::Size() const {
  return size_;
}

BufferType VulkanStaticBuffer::Type() const {
  return BUFFER_TYPE_STATIC;
}

void VulkanStaticBuffer::Resize(size_t new_size) {
  core_->WaitGPU();
  VkBuffer new_buffer = VK_NULL_HANDLE;
  VmaAllocation new_allocation = VK_NULL_HANDLE;
  vulkan::ThrowIfFailed(
      core_->CreateBuffer(new_size, StaticUsage(core_), VMA_MEMORY_USAGE_GPU_ONLY, &new_buffer, &new_allocation),
      "Failed to resize Vulkan static buffer");
  core_->SingleTimeCommand([&](VkCommandBuffer command_buffer) {
    VkBufferCopy copy_region{};
    copy_region.size = std::min(size_, new_size);
    vkCmdCopyBuffer(command_buffer, buffer_, new_buffer, 1, &copy_region);
  });
  vmaDestroyBuffer(core_->Allocator(), buffer_, allocation_);
  buffer_ = new_buffer;
  allocation_ = new_allocation;
  size_ = new_size;
}

void VulkanStaticBuffer::UploadData(const void *data, size_t size, size_t offset) {
  CpuProfileScope upload_profile("static_upload");
  if (FrameProfile::active) {
    ++FrameProfile::active->counters["static_upload_calls"];
    FrameProfile::active->counters["static_upload_bytes"] += size;
  }
  core_->WaitGPU();
  auto staging_buffer = core_->RequestUploadStagingBuffer(size);
  void *mapped = nullptr;
  vmaMapMemory(core_->Allocator(), core_->UploadStagingAllocation(), &mapped);
  std::memcpy(mapped, data, size);
  vmaUnmapMemory(core_->Allocator(), core_->UploadStagingAllocation());
  core_->SingleTimeCommand([&](VkCommandBuffer command_buffer) {
    VkBufferCopy copy_region{};
    copy_region.size = size;
    copy_region.dstOffset = offset;
    vkCmdCopyBuffer(command_buffer, staging_buffer, buffer_, 1, &copy_region);
  });
}

void VulkanStaticBuffer::DownloadData(void *data, size_t size, size_t offset) {
  core_->WaitGPU();
  auto staging_buffer = core_->RequestDownloadStagingBuffer(size);
  core_->SingleTimeCommand([&](VkCommandBuffer command_buffer) {
    VkBufferCopy copy_region{};
    copy_region.size = size;
    copy_region.srcOffset = offset;
    vkCmdCopyBuffer(command_buffer, buffer_, staging_buffer, 1, &copy_region);
  });
  void *mapped = nullptr;
  vmaMapMemory(core_->Allocator(), core_->DownloadStagingAllocation(), &mapped);
  std::memcpy(data, mapped, size);
  vmaUnmapMemory(core_->Allocator(), core_->DownloadStagingAllocation());
}

VkBuffer VulkanStaticBuffer::Buffer() const {
  return buffer_;
}

VkDeviceAddress VulkanStaticBuffer::DeviceAddress() const {
  return core_->BufferAddress(buffer_);
}

VulkanDynamicBuffer::VulkanDynamicBuffer(VulkanCore *core, size_t size) : core_(core), size_(size) {
  vulkan::ThrowIfFailed(core_->CreateBuffer(size, DynamicStagingUsage(core), VMA_MEMORY_USAGE_CPU_TO_GPU,
                                            &staging_buffer_, &staging_allocation_),
                        "Failed to create Vulkan dynamic staging buffer");
  buffers_.resize(core_->FramesInFlight(), VK_NULL_HANDLE);
  allocations_.resize(core_->FramesInFlight(), VK_NULL_HANDLE);
  buffer_sizes_.resize(core_->FramesInFlight(), size);
  for (size_t i = 0; i < buffers_.size(); ++i) {
    vulkan::ThrowIfFailed(
        core_->CreateBuffer(size, StaticUsage(core), VMA_MEMORY_USAGE_GPU_ONLY, &buffers_[i], &allocations_[i]),
        "Failed to create Vulkan dynamic device buffer");
  }
}

VulkanDynamicBuffer::~VulkanDynamicBuffer() {
  core_->WaitUploads();
  for (size_t i = 0; i < buffers_.size(); ++i)
    vmaDestroyBuffer(core_->Allocator(), buffers_[i], allocations_[i]);
  vmaDestroyBuffer(core_->Allocator(), staging_buffer_, staging_allocation_);
}

size_t VulkanDynamicBuffer::Size() const {
  return size_;
}

BufferType VulkanDynamicBuffer::Type() const {
  return BUFFER_TYPE_DYNAMIC;
}

void VulkanDynamicBuffer::Resize(size_t new_size) {
  core_->WaitUploads();
  VkBuffer new_buffer = VK_NULL_HANDLE;
  VmaAllocation new_allocation = VK_NULL_HANDLE;
  vulkan::ThrowIfFailed(core_->CreateBuffer(new_size, DynamicStagingUsage(core_), VMA_MEMORY_USAGE_CPU_TO_GPU,
                                            &new_buffer, &new_allocation),
                        "Failed to resize Vulkan dynamic staging buffer");
  void *new_data = nullptr;
  void *old_data = nullptr;
  vmaMapMemory(core_->Allocator(), new_allocation, &new_data);
  vmaMapMemory(core_->Allocator(), staging_allocation_, &old_data);
  std::memcpy(new_data, old_data, std::min(new_size, size_));
  vmaUnmapMemory(core_->Allocator(), staging_allocation_);
  vmaUnmapMemory(core_->Allocator(), new_allocation);
  vmaDestroyBuffer(core_->Allocator(), staging_buffer_, staging_allocation_);
  staging_buffer_ = new_buffer;
  staging_allocation_ = new_allocation;
  size_ = new_size;
}

void VulkanDynamicBuffer::UploadData(const void *data, size_t size, size_t offset) {
  // The staging memory is also the source of the last submitted upload.
  core_->WaitUploads();
  void *mapped = nullptr;
  vmaMapMemory(core_->Allocator(), staging_allocation_, &mapped);
  std::memcpy(static_cast<uint8_t *>(mapped) + offset, data, size);
  vmaUnmapMemory(core_->Allocator(), staging_allocation_);
}

void VulkanDynamicBuffer::DownloadData(void *data, size_t size, size_t offset) {
  void *mapped = nullptr;
  vmaMapMemory(core_->Allocator(), staging_allocation_, &mapped);
  std::memcpy(data, static_cast<uint8_t *>(mapped) + offset, size);
  vmaUnmapMemory(core_->Allocator(), staging_allocation_);
}

VkBuffer VulkanDynamicBuffer::Buffer() const {
  return buffers_[core_->CurrentFrame()];
}

VkDeviceAddress VulkanDynamicBuffer::DeviceAddress() const {
  return core_->BufferAddress(staging_buffer_);
}

void VulkanDynamicBuffer::TransferData(VkCommandBuffer cmd_buffer) {
  size_t frame = core_->CurrentFrame();
  if (buffer_sizes_[frame] != size_) {
    vmaDestroyBuffer(core_->Allocator(), buffers_[frame], allocations_[frame]);
    vulkan::ThrowIfFailed(core_->CreateBuffer(size_, StaticUsage(core_), VMA_MEMORY_USAGE_GPU_ONLY, &buffers_[frame],
                                              &allocations_[frame]),
                          "Failed to resize Vulkan dynamic device buffer");
    buffer_sizes_[frame] = size_;
  }
  VkBufferCopy copy_region{};
  copy_region.size = size_;
  vkCmdCopyBuffer(cmd_buffer, staging_buffer_, buffers_[frame], 1, &copy_region);
}

#if defined(LONGMARCH_CUDA_RUNTIME)
VulkanCUDABuffer::VulkanCUDABuffer(VulkanCore *core, size_t size) : core_(core), size_(size) {
  auto usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
               VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
               VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  if (core_->DeviceRayTracingSupport() || core_->DeviceRayQuerySupport()) {
    usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
             VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
  }
  vulkan::CreateExternalBuffer(
      core_->Handle(),
      [core_ = this->core_](uint32_t type_filter, VkMemoryPropertyFlags properties) {
        return core_->FindMemoryType(type_filter, properties);
      },
      size, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer_, memory_);
  core_->ImportCudaExternalMemory(cuda_memory_, memory_, size);
  VkBufferDeviceAddressInfo buffer_device_address_info{};
  buffer_device_address_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
  buffer_device_address_info.buffer = buffer_;
  address_ = core_->Procedures().vkGetBufferDeviceAddressKHR(core_->Handle(), &buffer_device_address_info);
}

VulkanCUDABuffer::~VulkanCUDABuffer() {
  Reset();
}

void VulkanCUDABuffer::Reset() {
  cudaDestroyExternalMemory(cuda_memory_);
  if (memory_) {
    vkFreeMemory(core_->Handle(), memory_, nullptr);
    memory_ = VK_NULL_HANDLE;
  }
  if (buffer_) {
    vkDestroyBuffer(core_->Handle(), buffer_, nullptr);
    buffer_ = VK_NULL_HANDLE;
  }
}

size_t VulkanCUDABuffer::Size() const {
  return size_;
}

BufferType VulkanCUDABuffer::Type() const {
  return BUFFER_TYPE_STATIC;
}

void VulkanCUDABuffer::Resize(size_t new_size) {
  core_->WaitGPU();
  auto usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
               VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
               VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  if (core_->DeviceRayTracingSupport() || core_->DeviceRayQuerySupport()) {
    usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
             VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
  }

  VkBuffer new_buffer;
  VkDeviceMemory new_memory;
  vulkan::CreateExternalBuffer(
      core_->Handle(),
      [core_ = this->core_](uint32_t type_filter, VkMemoryPropertyFlags properties) {
        return core_->FindMemoryType(type_filter, properties);
      },
      new_size, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, new_buffer, new_memory);
  core_->SingleTimeCommand([&](VkCommandBuffer command_buffer) {
    VkBufferCopy copy_region{};
    copy_region.size = std::min(static_cast<size_t>(size_), new_size);
    vkCmdCopyBuffer(command_buffer, buffer_, new_buffer, 1, &copy_region);
  });
  Reset();
  buffer_ = new_buffer;
  memory_ = new_memory;
  size_ = new_size;
  core_->ImportCudaExternalMemory(cuda_memory_, new_memory, new_size);
  VkBufferDeviceAddressInfo buffer_device_address_info{};
  buffer_device_address_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
  buffer_device_address_info.buffer = buffer_;
  address_ = core_->Procedures().vkGetBufferDeviceAddressKHR(core_->Handle(), &buffer_device_address_info);
}

void VulkanCUDABuffer::UploadData(const void *data, size_t size, size_t offset) {
  core_->WaitGPU();
  auto staging_buffer = core_->RequestUploadStagingBuffer(size);
  void *mapped = nullptr;
  vmaMapMemory(core_->Allocator(), core_->UploadStagingAllocation(), &mapped);
  std::memcpy(mapped, data, size);
  vmaUnmapMemory(core_->Allocator(), core_->UploadStagingAllocation());
  core_->SingleTimeCommand([&](VkCommandBuffer command_buffer) {
    VkBufferCopy copy_region{};
    copy_region.size = size;
    copy_region.dstOffset = offset;
    vkCmdCopyBuffer(command_buffer, staging_buffer, buffer_, 1, &copy_region);
  });
}

void VulkanCUDABuffer::DownloadData(void *data, size_t size, size_t offset) {
  core_->WaitGPU();
  auto staging_buffer = core_->RequestDownloadStagingBuffer(size);
  core_->SingleTimeCommand([&](VkCommandBuffer command_buffer) {
    VkBufferCopy copy_region{};
    copy_region.size = size;
    copy_region.srcOffset = offset;
    vkCmdCopyBuffer(command_buffer, buffer_, staging_buffer, 1, &copy_region);
  });
  void *mapped = nullptr;
  vmaMapMemory(core_->Allocator(), core_->DownloadStagingAllocation(), &mapped);
  std::memcpy(data, mapped, size);
  vmaUnmapMemory(core_->Allocator(), core_->DownloadStagingAllocation());
}

VkBuffer VulkanCUDABuffer::Buffer() const {
  return buffer_;
}

VkDeviceAddress VulkanCUDABuffer::DeviceAddress() const {
  return address_;
}

void VulkanCUDABuffer::GetCUDAMemoryPointer(void **ptr) {
  cudaExternalMemoryBufferDesc externalMemBufferDesc = {};
  externalMemBufferDesc.offset = 0;
  externalMemBufferDesc.size = size_;
  externalMemBufferDesc.flags = 0;
  cudaExternalMemoryGetMappedBuffer(ptr, cuda_memory_, &externalMemBufferDesc);
}

#endif

}  // namespace grassland::graphics::backend
