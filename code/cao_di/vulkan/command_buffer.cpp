#include "cao_di/vulkan/command_buffer.h"

namespace CD::vulkan {

CommandBuffer::CommandBuffer(const struct CommandPool *command_pool, VkCommandBuffer command_buffer)
    : command_pool_(command_pool), command_buffer_(command_buffer) {
}

CommandBuffer::~CommandBuffer() {
  vkFreeCommandBuffers(command_pool_->Device()->Handle(), command_pool_->Handle(), 1, &command_buffer_);
}
}  // namespace CD::vulkan
