#include "grassland/graphics/backend/vulkan/vulkan_command_context.h"

#include "grassland/graphics/backend/vulkan/vulkan_buffer.h"
#include "grassland/graphics/backend/vulkan/vulkan_image.h"
#include "grassland/graphics/backend/vulkan/vulkan_program.h"
#include "grassland/graphics/backend/vulkan/vulkan_window.h"

namespace grassland::graphics::backend {

VulkanCommandContext::VulkanCommandContext(VulkanCore *core) : core_(core) {
}

void VulkanCommandContext::BindColorTargets(
    const std::vector<Image *> &images) {
  color_targets_.resize(images.size());
  for (size_t i = 0; i < images.size(); ++i) {
    color_targets_[i] = dynamic_cast<VulkanImage *>(images[i]);
  }
}

void VulkanCommandContext::BindDepthTarget(Image *image) {
  depth_target_ = dynamic_cast<VulkanImage *>(image);
}

void VulkanCommandContext::BindVertexBuffers(
    const std::vector<Buffer *> &buffers) {
  vertex_buffers_.resize(buffers.size());
  for (size_t i = 0; i < buffers.size(); ++i) {
    auto vk_buffer = dynamic_cast<VulkanBuffer *>(buffers[i]);
    vertex_buffers_[i] = vk_buffer;
    if (vk_buffer) {
      auto dynamic_buffer = dynamic_cast<VulkanDynamicBuffer *>(vk_buffer);
      if (dynamic_buffer) {
        dynamic_buffers_.insert(dynamic_buffer);
      }
    }
  }
}

void VulkanCommandContext::BindIndexBuffer(Buffer *buffer) {
  index_buffer_ = dynamic_cast<VulkanBuffer *>(buffer);
  if (index_buffer_) {
    auto dynamic_buffer = dynamic_cast<VulkanDynamicBuffer *>(index_buffer_);
    if (dynamic_buffer) {
      dynamic_buffers_.insert(dynamic_buffer);
    }
  }
}

void VulkanCommandContext::BindProgram(Program *program) {
  program_ = dynamic_cast<VulkanProgram *>(program);
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
                                          uint32_t vertex_offset,
                                          uint32_t first_instance) {
  commands_.push_back(std::make_unique<VulkanCmdDrawIndexed>(
      program_, vertex_buffers_, index_buffer_, color_targets_, depth_target_,
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
                                             VkAccessFlags access) {
  if (image_states_.count(image) == 0) {
    image_states_[image] = {VK_IMAGE_LAYOUT_GENERAL,
                            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_ACCESS_NONE};
  }
  auto &state = image_states_[image];
  vulkan::TransitImageLayout(cmd_buffer, image, state.layout, layout,
                             state.stage, stage, state.access, access);
  state = {layout, stage, access};
}

}  // namespace grassland::graphics::backend
