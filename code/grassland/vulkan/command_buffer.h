#pragma once
#include "grassland/vulkan/command_pool.h"
#include "grassland/vulkan/vulkan_util.h"

namespace grassland::vulkan {
class CommandBuffer {
 public:
  CommandBuffer(const class CommandPool *command_pool,
                VkCommandBuffer command_buffer);

  ~CommandBuffer();

  VkCommandBuffer Handle() const {
    return command_buffer_;
  }

  const class CommandPool *CommandPool() const {
    return command_pool_;
  }

  operator VkCommandBuffer() const {
    return command_buffer_;
  }

 private:
  const class CommandPool *command_pool_;
  VkCommandBuffer command_buffer_{};
};
}  // namespace grassland::vulkan
