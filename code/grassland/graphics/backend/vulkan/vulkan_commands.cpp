#include "grassland/graphics/backend/vulkan/vulkan_commands.h"

#include "grassland/graphics/backend/vulkan/vulkan_command_context.h"
#include "grassland/graphics/backend/vulkan/vulkan_image.h"
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
