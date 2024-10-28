#pragma once
#include "grassland/vulkan/device.h"
#include "grassland/vulkan/vulkan_util.h"

namespace grassland::vulkan {
class CommandPool {
 public:
  CommandPool(const class Device *device, VkCommandPool command_pool);

  ~CommandPool();

  VkCommandPool Handle() const {
    return command_pool_;
  }

  const class Device *Device() const {
    return device_;
  }

  VkResult AllocateCommandBuffer(
      VkCommandBufferLevel level,
      double_ptr<CommandBuffer> pp_command_buffer) const;

  VkResult AllocateCommandBuffer(
      double_ptr<CommandBuffer> pp_command_buffer) const;

  VkResult SingleTimeCommands(
      Queue *queue,
      const std::function<void(VkCommandBuffer)> &tasks) const;

  VkResult SingleTimeCommands(
      const std::function<void(VkCommandBuffer)> &tasks) const;

 private:
  const class Device *device_;
  VkCommandPool command_pool_{};
};
}  // namespace grassland::vulkan
