#include "grassland/graphics/backend/vulkan/vulkan_image.h"

namespace CD::graphics::backend {

VulkanImage::VulkanImage(VulkanCore *core, int width, int height, ImageFormat format) : core_(core), format_(format) {
  VkExtent2D extent;
  extent.width = width;
  extent.height = height;
  core_->Device()->CreateImage(ImageFormatToVkFormat(format), extent, &image_);
}

Extent2D VulkanImage::Extent() const {
  auto extent = image_->Extent();
  return {extent.width, extent.height};
}

ImageFormat VulkanImage::Format() const {
  return format_;
}

void VulkanImage::UploadData(const void *data) const {
  auto extent = image_->Extent();
  auto pixel_size = static_cast<size_t>(PixelSize(format_));
  std::unique_ptr<vulkan::Buffer> staging_buffer;
  core_->Device()->CreateBuffer(pixel_size * extent.width * extent.height, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                VMA_MEMORY_USAGE_CPU_ONLY, &staging_buffer);
  std::memcpy(staging_buffer->Map(), data, pixel_size * extent.width * extent.height);
  staging_buffer->Unmap();
  core_->SingleTimeCommand([&](VkCommandBuffer command_buffer) {
    VkImageAspectFlagBits aspect = IsDepthFormat(format_) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    vulkan::TransitImageLayout(command_buffer, image_->Handle(), VK_IMAGE_LAYOUT_GENERAL,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                               VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_ACCESS_TRANSFER_WRITE_BIT, aspect);
    VkImageSubresourceLayers subresource{};
    subresource.aspectMask = aspect;
    subresource.mipLevel = 0;
    subresource.baseArrayLayer = 0;
    subresource.layerCount = 1;
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource = subresource;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {extent.width, extent.height, 1};
    vkCmdCopyBufferToImage(command_buffer, staging_buffer->Handle(), image_->Handle(),
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    vulkan::TransitImageLayout(command_buffer, image_->Handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, 0, aspect);
  });
}

void VulkanImage::DownloadData(void *data) const {
  auto extent = image_->Extent();
  auto pixel_size = static_cast<size_t>(PixelSize(format_));
  std::unique_ptr<vulkan::Buffer> staging_buffer;
  core_->Device()->CreateBuffer(pixel_size * extent.width * extent.height, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                VMA_MEMORY_USAGE_CPU_ONLY, &staging_buffer);
  core_->SingleTimeCommand([&](VkCommandBuffer command_buffer) {
    VkImageAspectFlagBits aspect = IsDepthFormat(format_) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    vulkan::TransitImageLayout(command_buffer, image_->Handle(), VK_IMAGE_LAYOUT_GENERAL,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                               VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_ACCESS_TRANSFER_READ_BIT, aspect);
    VkImageSubresourceLayers subresource{};
    subresource.aspectMask = aspect;
    subresource.mipLevel = 0;
    subresource.baseArrayLayer = 0;
    subresource.layerCount = 1;
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource = subresource;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {extent.width, extent.height, 1};
    vkCmdCopyImageToBuffer(command_buffer, image_->Handle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           staging_buffer->Handle(), 1, &region);
    vulkan::TransitImageLayout(command_buffer, image_->Handle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, VK_ACCESS_TRANSFER_READ_BIT, 0, aspect);
  });

  std::memcpy(data, staging_buffer->Map(), pixel_size * extent.width * extent.height);
  staging_buffer->Unmap();
}

}  // namespace CD::graphics::backend
