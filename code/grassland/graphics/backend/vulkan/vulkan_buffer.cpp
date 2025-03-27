#include "grassland/graphics/backend/vulkan/vulkan_buffer.h"

namespace grassland::graphics::backend {

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
  vulkan::SingleTimeCommand(core_->TransferQueue(), core_->TransferCommandPool(), [&](VkCommandBuffer command_buffer) {
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
  vulkan::SingleTimeCommand(core_->TransferQueue(), core_->TransferCommandPool(), [&](VkCommandBuffer command_buffer) {
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
  vulkan::SingleTimeCommand(core_->TransferQueue(), core_->TransferCommandPool(), [&](VkCommandBuffer command_buffer) {
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

vulkan::Buffer *VulkanStaticBuffer::InstantBuffer() const {
  return buffer_.get();
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

vulkan::Buffer *VulkanDynamicBuffer::InstantBuffer() const {
  return staging_buffer_.get();
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

}  // namespace grassland::graphics::backend
