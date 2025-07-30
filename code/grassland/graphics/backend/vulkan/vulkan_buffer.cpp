#include "grassland/graphics/backend/vulkan/vulkan_buffer.h"

namespace grassland::graphics::backend {

VulkanBufferRange::VulkanBufferRange(const BufferRange &range)
    : buffer(dynamic_cast<VulkanBuffer *>(range.buffer)), offset(range.offset), size(range.size) {
}

VulkanStaticBuffer::VulkanStaticBuffer(VulkanCore *core, size_t size) : core_(core) {
  auto usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
               VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
               VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  if (core_->DeviceRayTracingSupport()) {
    usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
             VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
  }
  core_->Device()->CreateBuffer(size, usage, VMA_MEMORY_USAGE_GPU_ONLY, &buffer_);
}

VulkanStaticBuffer::~VulkanStaticBuffer() {
  buffer_.reset();
}

size_t VulkanStaticBuffer::Size() const {
  return buffer_->Size();
}

BufferType VulkanStaticBuffer::Type() const {
  return BUFFER_TYPE_STATIC;
}

void VulkanStaticBuffer::Resize(size_t new_size) {
  core_->WaitGPU();
  std::unique_ptr<vulkan::Buffer> new_buffer;
  auto usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
               VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
               VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  if (core_->DeviceRayTracingSupport()) {
    usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
             VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
  }
  core_->Device()->CreateBuffer(new_size, usage, VMA_MEMORY_USAGE_GPU_ONLY, &new_buffer);
  core_->SingleTimeCommand([&](VkCommandBuffer command_buffer) {
    VkBufferCopy copy_region{};
    copy_region.size = std::min(static_cast<size_t>(buffer_->Size()), new_size);
    vkCmdCopyBuffer(command_buffer, buffer_->Handle(), new_buffer->Handle(), 1, &copy_region);
  });
  buffer_.reset();
  buffer_ = std::move(new_buffer);
}

void VulkanStaticBuffer::UploadData(const void *data, size_t size, size_t offset) {
  core_->WaitGPU();
  std::unique_ptr<vulkan::Buffer> staging_buffer;
  core_->Device()->CreateBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY, &staging_buffer);
  std::memcpy(staging_buffer->Map(), data, size);
  staging_buffer->Unmap();
  core_->SingleTimeCommand([&](VkCommandBuffer command_buffer) {
    VkBufferCopy copy_region{};
    copy_region.size = size;
    copy_region.dstOffset = offset;
    vkCmdCopyBuffer(command_buffer, staging_buffer->Handle(), buffer_->Handle(), 1, &copy_region);
  });
}

void VulkanStaticBuffer::DownloadData(void *data, size_t size, size_t offset) {
  core_->WaitGPU();
  std::unique_ptr<vulkan::Buffer> staging_buffer;
  core_->Device()->CreateBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_ONLY, &staging_buffer);
  core_->SingleTimeCommand([&](VkCommandBuffer command_buffer) {
    VkBufferCopy copy_region{};
    copy_region.size = size;
    copy_region.srcOffset = offset;
    vkCmdCopyBuffer(command_buffer, buffer_->Handle(), staging_buffer->Handle(), 1, &copy_region);
  });
  std::memcpy(data, staging_buffer->Map(), size);
  staging_buffer->Unmap();
}

VkBuffer VulkanStaticBuffer::Buffer() const {
  return buffer_->Handle();
}

VkDeviceAddress VulkanStaticBuffer::DeviceAddress() const {
  return buffer_->GetDeviceAddress();
}

VulkanDynamicBuffer::VulkanDynamicBuffer(VulkanCore *core, size_t size) : core_(core) {
  auto usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  if (core_->DeviceRayTracingSupport()) {
    usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
             VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
  }
  core_->Device()->CreateBuffer(size, usage, VMA_MEMORY_USAGE_CPU_TO_GPU, &staging_buffer_);

  usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;

  buffers_.resize(core_->FramesInFlight());

  for (size_t i = 0; i < buffers_.size(); ++i) {
    core_->Device()->CreateBuffer(size, usage, VMA_MEMORY_USAGE_GPU_ONLY, &buffers_[i]);
  }
}

VulkanDynamicBuffer::~VulkanDynamicBuffer() {
  buffers_.clear();
  staging_buffer_.reset();
}

size_t VulkanDynamicBuffer::Size() const {
  return staging_buffer_->Size();
}

BufferType VulkanDynamicBuffer::Type() const {
  return BUFFER_TYPE_DYNAMIC;
}

void VulkanDynamicBuffer::Resize(size_t new_size) {
  std::unique_ptr<vulkan::Buffer> new_buffer;
  auto usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  if (core_->DeviceRayTracingSupport()) {
    usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
             VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
  }
  core_->Device()->CreateBuffer(new_size, usage, VMA_MEMORY_USAGE_CPU_TO_GPU, &new_buffer);
  std::memcpy(new_buffer->Map(), staging_buffer_->Map(),
              std::min(new_size, static_cast<size_t>(staging_buffer_->Size())));
  new_buffer->Unmap();
  staging_buffer_->Unmap();
  staging_buffer_.reset();
  staging_buffer_ = std::move(new_buffer);
}

void VulkanDynamicBuffer::UploadData(const void *data, size_t size, size_t offset) {
  std::memcpy(static_cast<uint8_t *>(staging_buffer_->Map()) + offset, data, size);
  staging_buffer_->Unmap();
}

void VulkanDynamicBuffer::DownloadData(void *data, size_t size, size_t offset) {
  std::memcpy(data, static_cast<uint8_t *>(staging_buffer_->Map()) + offset, size);
  staging_buffer_->Unmap();
}

VkBuffer VulkanDynamicBuffer::Buffer() const {
  return buffers_[core_->CurrentFrame()]->Handle();
}

VkDeviceAddress VulkanDynamicBuffer::DeviceAddress() const {
  return staging_buffer_->GetDeviceAddress();
}

void VulkanDynamicBuffer::TransferData(VkCommandBuffer cmd_buffer) {
  if (buffers_[core_->CurrentFrame()]->Size() != staging_buffer_->Size()) {
    buffers_[core_->CurrentFrame()].reset();
    auto usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    core_->Device()->CreateBuffer(staging_buffer_->Size(), usage, VMA_MEMORY_USAGE_GPU_ONLY,
                                  &buffers_[core_->CurrentFrame()]);
  }

  VkBufferCopy copy_region{};
  copy_region.size = staging_buffer_->Size();

  vkCmdCopyBuffer(cmd_buffer, staging_buffer_->Handle(), buffers_[core_->CurrentFrame()]->Handle(), 1, &copy_region);
}

#if defined(LONGMARCH_CUDA_RUNTIME)
VulkanCUDABuffer::VulkanCUDABuffer(VulkanCore *core, size_t size) : core_(core), size_(size) {
  auto usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
               VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
               VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  if (core_->DeviceRayTracingSupport()) {
    usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
             VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
  }
  vulkan::CreateExternalBuffer(
      core_->Device()->Handle(),
      [core_ = this->core_](uint32_t type_filter, VkMemoryPropertyFlags properties) {
        return core_->FindMemoryType(type_filter, properties);
      },
      size, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer_, memory_);
  core_->ImportCudaExternalMemory(cuda_memory_, memory_, size);
  VkBufferDeviceAddressInfo buffer_device_address_info{};
  buffer_device_address_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
  buffer_device_address_info.buffer = buffer_;
  address_ =
      core_->Device()->Procedures().vkGetBufferDeviceAddressKHR(core_->Device()->Handle(), &buffer_device_address_info);
}

VulkanCUDABuffer::~VulkanCUDABuffer() {
  Reset();
}

void VulkanCUDABuffer::Reset() {
  cudaDestroyExternalMemory(cuda_memory_);
  if (memory_) {
    vkFreeMemory(core_->Device()->Handle(), memory_, nullptr);
    memory_ = VK_NULL_HANDLE;
  }
  if (buffer_) {
    vkDestroyBuffer(core_->Device()->Handle(), buffer_, nullptr);
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
  if (core_->DeviceRayTracingSupport()) {
    usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
             VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
  }
  VkBuffer new_buffer;
  VkDeviceMemory new_memory;
  vulkan::CreateExternalBuffer(
      core_->Device()->Handle(),
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
  address_ =
      core_->Device()->Procedures().vkGetBufferDeviceAddressKHR(core_->Device()->Handle(), &buffer_device_address_info);
}

void VulkanCUDABuffer::UploadData(const void *data, size_t size, size_t offset) {
  core_->WaitGPU();
  std::unique_ptr<vulkan::Buffer> staging_buffer;
  core_->Device()->CreateBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY, &staging_buffer);
  std::memcpy(staging_buffer->Map(), data, size);
  staging_buffer->Unmap();
  core_->SingleTimeCommand([&](VkCommandBuffer command_buffer) {
    VkBufferCopy copy_region{};
    copy_region.size = size;
    copy_region.dstOffset = offset;
    vkCmdCopyBuffer(command_buffer, staging_buffer->Handle(), buffer_, 1, &copy_region);
  });
}

void VulkanCUDABuffer::DownloadData(void *data, size_t size, size_t offset) {
  core_->WaitGPU();
  std::unique_ptr<vulkan::Buffer> staging_buffer;
  core_->Device()->CreateBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_ONLY, &staging_buffer);
  core_->SingleTimeCommand([&](VkCommandBuffer command_buffer) {
    VkBufferCopy copy_region{};
    copy_region.size = size;
    copy_region.srcOffset = offset;
    vkCmdCopyBuffer(command_buffer, buffer_, staging_buffer->Handle(), 1, &copy_region);
  });
  std::memcpy(data, staging_buffer->Map(), size);
  staging_buffer->Unmap();
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
