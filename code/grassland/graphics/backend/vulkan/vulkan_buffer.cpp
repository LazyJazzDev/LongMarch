#include "grassland/graphics/backend/vulkan/vulkan_buffer.h"

namespace grassland::graphics::backend {

VulkanStaticBuffer::VulkanStaticBuffer(VulkanCore *core, size_t size)
    : core_(core) {
  auto usage =
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
      VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  core_->Device()->CreateBuffer(size, usage, VMA_MEMORY_USAGE_GPU_ONLY,
                                &buffer_);
}

VulkanStaticBuffer::~VulkanStaticBuffer() {
  buffer_.reset();
}

void VulkanStaticBuffer::Resize(size_t new_size) {
  core_->WaitGPU();
  std::unique_ptr<vulkan::Buffer> new_buffer;
  auto usage =
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
      VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  core_->Device()->CreateBuffer(new_size, usage, VMA_MEMORY_USAGE_GPU_ONLY,
                                &new_buffer);
  vulkan::SingleTimeCommand(
      core_->TransferQueue(), core_->TransferCommandPool(),
      [&](VkCommandBuffer command_buffer) {
        VkBufferCopy copy_region{};
        copy_region.size = std::min(buffer_->Size(), new_size);
        vkCmdCopyBuffer(command_buffer, buffer_->Handle(), new_buffer->Handle(),
                        1, &copy_region);
      });
  buffer_.reset();
  buffer_ = std::move(new_buffer);
}

void VulkanStaticBuffer::UploadData(const void *data,
                                    size_t size,
                                    size_t offset) {
  core_->WaitGPU();
  std::unique_ptr<vulkan::Buffer> staging_buffer;
  core_->Device()->CreateBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                VMA_MEMORY_USAGE_CPU_ONLY, &staging_buffer);
  std::memcpy(staging_buffer->Map(), data, size);
  staging_buffer->Unmap();
  vulkan::SingleTimeCommand(
      core_->TransferQueue(), core_->TransferCommandPool(),
      [&](VkCommandBuffer command_buffer) {
        VkBufferCopy copy_region{};
        copy_region.size = size;
        copy_region.dstOffset = offset;
        vkCmdCopyBuffer(command_buffer, staging_buffer->Handle(),
                        buffer_->Handle(), 1, &copy_region);
      });
}

void VulkanStaticBuffer::DownloadData(void *data, size_t size, size_t offset) {
  core_->WaitGPU();
  std::unique_ptr<vulkan::Buffer> staging_buffer;
  core_->Device()->CreateBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                VMA_MEMORY_USAGE_CPU_ONLY, &staging_buffer);
  vulkan::SingleTimeCommand(
      core_->TransferQueue(), core_->TransferCommandPool(),
      [&](VkCommandBuffer command_buffer) {
        VkBufferCopy copy_region{};
        copy_region.size = size;
        copy_region.srcOffset = offset;
        vkCmdCopyBuffer(command_buffer, buffer_->Handle(),
                        staging_buffer->Handle(), 1, &copy_region);
      });
  std::memcpy(data, staging_buffer->Map(), size);
  staging_buffer->Unmap();
}

}  // namespace grassland::graphics::backend
