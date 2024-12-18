#include "grassland/graphics/backend/vulkan/vulkan_commands.h"

#include "grassland/graphics/backend/vulkan/vulkan_command_context.h"
#include "grassland/graphics/backend/vulkan/vulkan_image.h"
#include "vulkan_buffer.h"
#include "vulkan_program.h"
#include "vulkan_window.h"

namespace grassland::graphics::backend {

VulkanCmdClearImage::VulkanCmdClearImage(VulkanImage *image,
                                         const ClearValue &clear_value)
    : image_(image), clear_value_(clear_value) {
}

void VulkanCmdClearImage::CompileCommand(VulkanCommandContext *context,
                                         VkCommandBuffer command_buffer) {
  context->RequireImageState(command_buffer, image_->Image()->Handle(),
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_ACCESS_TRANSFER_WRITE_BIT);
  if (!vulkan::IsDepthFormat(image_->Image()->Format())) {
    VkClearColorValue clear_value;
    clear_value.float32[0] = clear_value_.color.r;
    clear_value.float32[1] = clear_value_.color.g;
    clear_value.float32[2] = clear_value_.color.b;
    clear_value.float32[3] = clear_value_.color.a;
    VkImageSubresourceRange subresource_range;
    subresource_range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresource_range.baseMipLevel = 0;
    subresource_range.levelCount = 1;
    subresource_range.baseArrayLayer = 0;
    subresource_range.layerCount = 1;
    vkCmdClearColorImage(command_buffer, image_->Image()->Handle(),
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear_value, 1,
                         &subresource_range);
  } else {
    VkClearDepthStencilValue clear_value;
    clear_value.depth = clear_value_.depth.depth;
    clear_value.stencil = 0;
    VkImageSubresourceRange subresource_range;
    subresource_range.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    subresource_range.baseMipLevel = 0;
    subresource_range.levelCount = 1;
    subresource_range.baseArrayLayer = 0;
    subresource_range.layerCount = 1;
    vkCmdClearDepthStencilImage(command_buffer, image_->Image()->Handle(),
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                &clear_value, 1, &subresource_range);
  }
}

VulkanCmdSetViewport::VulkanCmdSetViewport(const Viewport &viewport)
    : viewport_(viewport) {
}

void VulkanCmdSetViewport::CompileCommand(VulkanCommandContext *context,
                                          VkCommandBuffer command_buffer) {
  VkViewport viewport{};
  viewport.x = viewport_.x;
  viewport.y = viewport_.y;
  viewport.width = viewport_.width;
  viewport.height = viewport_.height;
  viewport.minDepth = viewport_.min_depth;
  viewport.maxDepth = viewport_.max_depth;
  vkCmdSetViewport(command_buffer, 0, 1, &viewport);
}

VulkanCmdSetScissor::VulkanCmdSetScissor(const Scissor &scissor)
    : scissor_(scissor) {
}

void VulkanCmdSetScissor::CompileCommand(VulkanCommandContext *context,
                                         VkCommandBuffer command_buffer) {
  VkRect2D scissor{};
  scissor.offset = {scissor_.offset.x, scissor_.offset.y};
  scissor.extent = {scissor_.extent.width, scissor_.extent.height};
  vkCmdSetScissor(command_buffer, 0, 1, &scissor);
}

VulkanCmdDrawIndexed::VulkanCmdDrawIndexed(
    VulkanProgram *program,
    const std::vector<VulkanBuffer *> &vertex_buffers,
    VulkanBuffer *index_buffer,
    const std::vector<VulkanImage *> &color_targets,
    VulkanImage *depth_target,
    uint32_t index_count,
    uint32_t instance_count,
    uint32_t first_index,
    uint32_t vertex_offset,
    uint32_t first_instance)
    : program_(program),
      vertex_buffers_({}),
      index_buffer_(index_buffer),
      color_targets_({}),
      depth_target_(nullptr),
      index_count_(index_count),
      instance_count_(instance_count),
      first_index_(first_index),
      vertex_offset_(vertex_offset),
      first_instance_(first_instance) {
  vertex_buffers_.resize(program_->NumInputBindings());
  for (size_t i = 0; i < vertex_buffers_.size(); ++i) {
    vertex_buffers_[i] = vertex_buffers[i];
  }
  color_targets_.resize(
      program_->PipelineSettings()->color_attachment_formats.size());
  for (size_t i = 0; i < color_targets_.size(); ++i) {
    color_targets_[i] = color_targets[i];
  }
  if (program_->PipelineSettings()->depth_attachment_format !=
      VK_FORMAT_UNDEFINED) {
    depth_target_ = depth_target;
  }
}

void VulkanCmdDrawIndexed::CompileCommand(VulkanCommandContext *context,
                                          VkCommandBuffer command_buffer) {
  std::vector<VkRenderingAttachmentInfo> color_attachment_infos;
  VkRenderingAttachmentInfo depth_attachment_info{};
  for (int i = 0; i < color_targets_.size(); i++) {
    auto &color_target = color_targets_[i];
    context->RequireImageState(command_buffer, color_target->Image()->Handle(),
                               VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                               VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                               VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);

    VkRenderingAttachmentInfo attachment_info{};
    attachment_info.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    attachment_info.pNext = nullptr;
    attachment_info.imageView = color_target->Image()->ImageView();
    attachment_info.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachment_info.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attachment_info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color_attachment_infos.push_back(attachment_info);
  }
  VkRenderingInfo rendering_info{};
  rendering_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
  rendering_info.pNext = nullptr;
  rendering_info.colorAttachmentCount = color_attachment_infos.size();
  rendering_info.pColorAttachments = color_attachment_infos.data();
  rendering_info.renderArea.offset = {0, 0};
  if (color_targets_.empty()) {
    rendering_info.renderArea.extent.width = depth_target_->Extent().width;
    rendering_info.renderArea.extent.height = depth_target_->Extent().height;
  } else {
    rendering_info.renderArea.extent.width = color_targets_[0]->Extent().width;
    rendering_info.renderArea.extent.height =
        color_targets_[0]->Extent().height;
  }
  rendering_info.layerCount = 1;
  if (depth_target_) {
    context->RequireImageState(command_buffer, depth_target_->Image()->Handle(),
                               VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                               VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                               VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
    depth_attachment_info.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depth_attachment_info.pNext = nullptr;
    depth_attachment_info.imageView = depth_target_->Image()->ImageView();
    depth_attachment_info.imageLayout =
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depth_attachment_info.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    depth_attachment_info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    rendering_info.pDepthAttachment = &depth_attachment_info;
  }
  context->Core()->Instance()->Procedures().vkCmdBeginRenderingKHR(
      command_buffer, &rendering_info);

  vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    program_->Pipeline()->Handle());

  if (!vertex_buffers_.empty()) {
    std::vector<VkBuffer> vertex_buffer_handles;
    std::vector<VkDeviceSize> vertex_buffer_offsets;
    for (int i = 0; i < vertex_buffers_.size(); i++) {
      auto &vertex_buffer = vertex_buffers_[i];
      vertex_buffer_handles.push_back(vertex_buffer->Buffer());
      vertex_buffer_offsets.push_back(0);
    }
    vkCmdBindVertexBuffers(command_buffer, 0, vertex_buffers_.size(),
                           vertex_buffer_handles.data(),
                           vertex_buffer_offsets.data());
  }

  vkCmdBindIndexBuffer(command_buffer, index_buffer_->Buffer(), 0,
                       VK_INDEX_TYPE_UINT32);

  vkCmdDrawIndexed(command_buffer, index_count_, instance_count_, first_index_,
                   vertex_offset_, first_instance_);

  context->Core()->Instance()->Procedures().vkCmdEndRenderingKHR(
      command_buffer);
}

VulkanCmdPresent::VulkanCmdPresent(VulkanWindow *window, VulkanImage *image)
    : image_(image), window_(window) {
}

void VulkanCmdPresent::CompileCommand(VulkanCommandContext *context,
                                      VkCommandBuffer command_buffer) {
  vulkan::TransitImageLayout(
      command_buffer, window_->CurrentImage(), VK_IMAGE_LAYOUT_UNDEFINED,
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
      VK_IMAGE_ASPECT_COLOR_BIT);

  context->RequireImageState(command_buffer, image_->Image()->Handle(),
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_ACCESS_TRANSFER_READ_BIT);

  auto image_extent = image_->Extent();
  auto window_extent = window_->SwapChain()->Extent();

  VkImageBlit blit{};
  blit.srcOffsets[0] = {0, 0, 0};
  blit.srcOffsets[1] = {static_cast<int32_t>(image_extent.width),
                        static_cast<int32_t>(image_extent.height), 1};
  blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  blit.srcSubresource.mipLevel = 0;
  blit.srcSubresource.baseArrayLayer = 0;
  blit.srcSubresource.layerCount = 1;
  blit.dstOffsets[0] = {0, 0, 0};
  blit.dstOffsets[1] = {static_cast<int32_t>(window_extent.width),
                        static_cast<int32_t>(window_extent.height), 1};
  blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  blit.dstSubresource.mipLevel = 0;
  blit.dstSubresource.baseArrayLayer = 0;
  blit.dstSubresource.layerCount = 1;
  vkCmdBlitImage(command_buffer, image_->Image()->Handle(),
                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, window_->CurrentImage(),
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                 VK_FILTER_LINEAR);

  vulkan::TransitImageLayout(
      command_buffer, window_->CurrentImage(),
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
      VK_ACCESS_TRANSFER_WRITE_BIT, 0, VK_IMAGE_ASPECT_COLOR_BIT);
}

}  // namespace grassland::graphics::backend
