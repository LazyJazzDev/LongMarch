#include "grassland/graphics/backend/vulkan/vulkan_command_context.h"

#include "grassland/graphics/backend/vulkan/vulkan_buffer.h"
#include "grassland/graphics/backend/vulkan/vulkan_image.h"
#include "grassland/graphics/backend/vulkan/vulkan_program.h"
#include "grassland/graphics/backend/vulkan/vulkan_window.h"

namespace grassland::graphics::backend {

VulkanCommandContext::VulkanCommandContext(VulkanCore *core) : core_(core) {
}

void VulkanCommandContext::CmdBindProgram(Program *program) {
  program_ = dynamic_cast<VulkanProgram *>(program);
  commands_.push_back(std::make_unique<VulkanCmdBindProgram>(program_));
}

void VulkanCommandContext::CmdBindVertexBuffers(
    uint32_t first_binding,
    const std::vector<Buffer *> &buffers,
    const std::vector<uint64_t> &offsets) {
  std::vector<VulkanBuffer *> vertex_buffers(buffers.size());
  for (size_t i = 0; i < buffers.size(); ++i) {
    vertex_buffers[i] = dynamic_cast<VulkanBuffer *>(buffers[i]);
    RecordDynamicBuffer(vertex_buffers[i]);
  }
  commands_.push_back(std::make_unique<VulkanCmdBindVertexBuffers>(
      first_binding, vertex_buffers, offsets));
}

void VulkanCommandContext::CmdBindIndexBuffer(Buffer *buffer, uint64_t offset) {
  auto index_buffer = dynamic_cast<VulkanBuffer *>(buffer);
  RecordDynamicBuffer(index_buffer);
  commands_.push_back(
      std::make_unique<VulkanCmdBindIndexBuffer>(index_buffer, offset));
}

void VulkanCommandContext::CmdBeginRendering(
    const std::vector<Image *> &color_targets,
    Image *depth_target) {
  std::vector<VulkanImage *> vk_color_targets(color_targets.size());
  VulkanImage *vk_depth_target{nullptr};

  for (size_t i = 0; i < color_targets.size(); ++i) {
    vk_color_targets[i] = dynamic_cast<VulkanImage *>(color_targets[i]);
  }

  if (depth_target) {
    vk_depth_target = dynamic_cast<VulkanImage *>(depth_target);
  }

  commands_.push_back(std::make_unique<VulkanCmdBeginRendering>(
      vk_color_targets, vk_depth_target));
}

void VulkanCommandContext::CmdBindResources(
    int slot,
    const std::vector<Buffer *> &buffers) {
  std::vector<VulkanBuffer *> vk_buffers(buffers.size());
  for (size_t i = 0; i < buffers.size(); ++i) {
    vk_buffers[i] = dynamic_cast<VulkanBuffer *>(buffers[i]);
    RecordDynamicBuffer(vk_buffers[i]);
  }
  commands_.push_back(std::make_unique<VulkanCmdBindResourceBuffers>(
      slot, vk_buffers, program_));
  required_pool_size_ += program_->DescriptorSetLayout(slot)->GetPoolSize();
  required_set_count_++;
}

void VulkanCommandContext::CmdEndRendering() {
  commands_.push_back(std::make_unique<VulkanCmdEndRendering>());
}

void VulkanCommandContext::CmdSetViewport(const Viewport &viewport) {
  commands_.push_back(std::make_unique<VulkanCmdSetViewport>(viewport));
}

void VulkanCommandContext::CmdSetScissor(const Scissor &scissor) {
  commands_.push_back(std::make_unique<VulkanCmdSetScissor>(scissor));
}

void VulkanCommandContext::CmdDrawIndexed(uint32_t index_count,
                                          uint32_t instance_count,
                                          uint32_t first_index,
                                          int32_t vertex_offset,
                                          uint32_t first_instance) {
  commands_.push_back(std::make_unique<VulkanCmdDrawIndexed>(
      index_count, instance_count, first_index, vertex_offset, first_instance));
}

void VulkanCommandContext::CmdClearImage(Image *image,
                                         const ClearValue &color) {
  commands_.push_back(std::make_unique<VulkanCmdClearImage>(
      dynamic_cast<VulkanImage *>(image), color));
}

void VulkanCommandContext::CmdPresent(Window *window, Image *image) {
  auto vulkan_window = dynamic_cast<VulkanWindow *>(window);
  auto vulkan_image = dynamic_cast<VulkanImage *>(image);
  commands_.push_back(
      std::make_unique<VulkanCmdPresent>(vulkan_window, vulkan_image));
  windows_.insert(vulkan_window);
}

void VulkanCommandContext::RequireImageState(VkCommandBuffer cmd_buffer,
                                             VkImage image,
                                             VkImageLayout layout,
                                             VkPipelineStageFlags stage,
                                             VkAccessFlags access,
                                             VkImageAspectFlags aspect) {
  if (image_states_.count(image) == 0) {
    image_states_[image] = {VK_IMAGE_LAYOUT_GENERAL,
                            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_ACCESS_NONE};
  }
  auto &state = image_states_[image];
  vulkan::TransitImageLayout(cmd_buffer, image, state.layout, layout,
                             state.stage, stage, state.access, access, aspect);
  state = {layout, stage, access, aspect};
}

void VulkanCommandContext::RecordDynamicBuffer(VulkanBuffer *buffer) {
  auto dynamic_buffer = dynamic_cast<VulkanDynamicBuffer *>(buffer);
  if (dynamic_buffer) {
    dynamic_buffers_.insert(dynamic_buffer);
  }
}

}  // namespace grassland::graphics::backend
