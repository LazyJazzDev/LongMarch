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

class VulkanCmdSetViewport : public VulkanCommand {
 public:
  VulkanCmdSetViewport(const Viewport &viewport);

  void CompileCommand(VulkanCommandContext *context,
                      VkCommandBuffer command_buffer) override;

 private:
  Viewport viewport_;
};

class VulkanCmdSetScissor : public VulkanCommand {
 public:
  VulkanCmdSetScissor(const Scissor &scissor);

  void CompileCommand(VulkanCommandContext *context,
                      VkCommandBuffer command_buffer) override;

 private:
  Scissor scissor_;
};

class VulkanCmdDrawIndexed : public VulkanCommand {
 public:
  VulkanCmdDrawIndexed(VulkanProgram *program,
                       const std::vector<VulkanBuffer *> &vertex_buffers,
                       VulkanBuffer *index_buffer,
                       const std::vector<VulkanImage *> &color_targets,
                       VulkanImage *depth_target,
                       uint32_t index_count,
                       uint32_t instance_count,
                       uint32_t first_index,
                       uint32_t vertex_offset,
                       uint32_t first_instance);

  void CompileCommand(VulkanCommandContext *context,
                      VkCommandBuffer command_buffer) override;

 private:
  VulkanProgram *program_;
  std::vector<VulkanBuffer *> vertex_buffers_;
  VulkanBuffer *index_buffer_;
  std::vector<VulkanImage *> color_targets_;
  VulkanImage *depth_target_;
  uint32_t index_count_;
  uint32_t instance_count_;
  uint32_t first_index_;
  uint32_t vertex_offset_;
  uint32_t first_instance_;
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
