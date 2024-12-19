#pragma once
#include "grassland/graphics/backend/vulkan/vulkan_buffer.h"
#include "grassland/graphics/backend/vulkan/vulkan_commands.h"
#include "grassland/graphics/backend/vulkan/vulkan_core.h"
#include "grassland/graphics/backend/vulkan/vulkan_util.h"

namespace grassland::graphics::backend {

class VulkanCommandContext : public CommandContext {
 public:
  VulkanCommandContext(VulkanCore *core);

  void BindColorTargets(const std::vector<Image *> &images) override;
  void BindDepthTarget(Image *image) override;
  void BindVertexBuffers(const std::vector<Buffer *> &buffers) override;
  void BindIndexBuffer(Buffer *buffer) override;
  void BindProgram(Program *program) override;

  void CmdSetViewport(const Viewport &viewport) override;
  void CmdSetScissor(const Scissor &scissor) override;
  void CmdDrawIndexed(uint32_t index_count,
                      uint32_t instance_count,
                      uint32_t first_index,
                      uint32_t vertex_offset,
                      uint32_t first_instance) override;
  void CmdClearImage(Image *image, const ClearValue &color) override;
  void CmdPresent(Window *window, Image *image) override;

  void RequireImageState(VkCommandBuffer cmd_buffer,
                         VkImage image,
                         VkImageLayout layout,
                         VkPipelineStageFlags stage,
                         VkAccessFlags access);

  VulkanCore *Core() const {
    return core_;
  }

 private:
  friend VulkanCore;
  VulkanCore *core_;

  VulkanProgram *program_{nullptr};
  std::vector<VulkanImage *> color_targets_;
  VulkanImage *depth_target_{nullptr};
  std::vector<VulkanBuffer *> vertex_buffers_;
  VulkanBuffer *index_buffer_{nullptr};

  std::vector<std::unique_ptr<VulkanCommand>> commands_;
  std::set<VulkanWindow *> windows_;

  std::set<VulkanDynamicBuffer *> dynamic_buffers_;

  struct ImageState {
    VkImageLayout layout;
    VkPipelineStageFlags stage;
    VkAccessFlags access;
  };

  std::map<VkImage, ImageState> image_states_;
};

}  // namespace grassland::graphics::backend
