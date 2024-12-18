#pragma once
#include "grassland/graphics/backend/vulkan/vulkan_core.h"

namespace grassland::graphics::backend {
class VulkanCommand {
 public:
  virtual ~VulkanCommand() = default;
  virtual void CompileCommand(VulkanCommandContext *context,
                              VkCommandBuffer command_buffer) = 0;
};

class VulkanCmdClearImage : public VulkanCommand {
 public:
  VulkanCmdClearImage(VulkanImage *image, const ClearValue &clear_value);

  void CompileCommand(VulkanCommandContext *context,
                      VkCommandBuffer command_buffer) override;

 private:
  VulkanImage *image_;
  ClearValue clear_value_;
};

class VulkanCmdPresent : public VulkanCommand {
 public:
  VulkanCmdPresent(VulkanWindow *window, VulkanImage *image);
  void CompileCommand(VulkanCommandContext *context,
                      VkCommandBuffer command_buffer) override;

 private:
  VulkanImage *image_;
  VulkanWindow *window_;
};

}  // namespace grassland::graphics::backend
