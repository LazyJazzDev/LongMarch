#pragma once
#include "grassland/graphics/backend/vulkan/vulkan_buffer.h"
#include "grassland/graphics/backend/vulkan/vulkan_commands.h"
#include "grassland/graphics/backend/vulkan/vulkan_core.h"
#include "grassland/graphics/backend/vulkan/vulkan_util.h"

namespace grassland::graphics::backend {

class VulkanCommandContext : public CommandContext {
 public:
  VulkanCommandContext(VulkanCore *core);

  void CmdBindProgram(Program *program) override;
  void CmdBindVertexBuffers(uint32_t first_binding,
                            const std::vector<Buffer *> &buffers,
                            const std::vector<uint64_t> &offsets) override;
  void CmdBindIndexBuffer(Buffer *buffer, uint64_t offset) override;
  void CmdBeginRendering(const std::vector<Image *> &color_targets,
                         Image *depth_target) override;
  void CmdBindResources(int slot,
                        const std::vector<Buffer *> &buffers) override;
  void CmdEndRendering() override;

  void CmdSetViewport(const Viewport &viewport) override;
  void CmdSetScissor(const Scissor &scissor) override;
  void CmdDrawIndexed(uint32_t index_count,
                      uint32_t instance_count,
                      uint32_t first_index,
                      int32_t vertex_offset,
                      uint32_t first_instance) override;
  void CmdClearImage(Image *image, const ClearValue &color) override;
  void CmdPresent(Window *window, Image *image) override;

  void RequireImageState(VkCommandBuffer cmd_buffer,
                         VkImage image,
                         VkImageLayout layout,
                         VkPipelineStageFlags stage,
                         VkAccessFlags access,
                         VkImageAspectFlags aspect = VK_IMAGE_ASPECT_DEPTH_BIT);

  VulkanCore *Core() const {
    return core_;
  }

  void RecordDynamicBuffer(VulkanBuffer *buffer);

  vulkan::DescriptorSet *AcquireDescriptorSet(
      VkDescriptorSetLayout set_layout) const;

 private:
  friend VulkanCore;
  VulkanCore *core_;

  VulkanProgram *program_{nullptr};

  std::vector<std::unique_ptr<VulkanCommand>> commands_;
  std::set<VulkanWindow *> windows_;

  std::set<VulkanDynamicBuffer *> dynamic_buffers_;

  struct ImageState {
    VkImageLayout layout;
    VkPipelineStageFlags stage;
    VkAccessFlags access;
    VkImageAspectFlags aspect;
  };

  std::map<VkImage, ImageState> image_states_;

  vulkan::DescriptorPoolSize required_pool_size_;
  int required_set_count_{0};
};

inline vulkan::DescriptorSet *VulkanCommandContext::AcquireDescriptorSet(
    VkDescriptorSetLayout set_layout) const {
  vulkan::DescriptorSet *descriptor_set = nullptr;
  core_->current_descriptor_pool_->AllocateDescriptorSet(set_layout,
                                                         &descriptor_set);
  core_->current_descriptor_set_queue_->push(descriptor_set);
  return descriptor_set;
}

}  // namespace grassland::graphics::backend
