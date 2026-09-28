#include "grassland/graphics/backend/vulkan/vulkan_image.h"

namespace grassland::graphics::backend {

VulkanImage::VulkanImage(VulkanCore *core, int width, int height, ImageFormat format) : core_(core), format_(format) {
  extent_ = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
  aspect_ =
      vulkan::IsDepthFormat(ImageFormatToVkFormat(format)) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
  vulkan::ThrowIfFailed(core_->CreateImage(ImageFormatToVkFormat(format), extent_, &image_, &image_view_, &allocation_),
                        "Failed to create Vulkan image");
}

VulkanImage::~VulkanImage() {
  vkDestroyImageView(core_->Handle(), image_view_, nullptr);
  vmaDestroyImage(core_->Allocator(), image_, allocation_);
}

Extent2D VulkanImage::Extent() const {
  return {extent_.width, extent_.height};
}

ImageFormat VulkanImage::Format() const {
  return format_;
}

void VulkanImage::UploadData(const void *data) const {
  auto extent = extent_;
  auto pixel_size = static_cast<size_t>(PixelSize(format_));
  VkBuffer staging_buffer = VK_NULL_HANDLE;
  VmaAllocation staging_allocation = VK_NULL_HANDLE;
  core_->CreateBuffer(pixel_size * extent.width * extent.height, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      VMA_MEMORY_USAGE_CPU_ONLY, &staging_buffer, &staging_allocation);
  void *mapped = nullptr;
  vmaMapMemory(core_->Allocator(), staging_allocation, &mapped);
  std::memcpy(mapped, data, pixel_size * extent.width * extent.height);
  vmaUnmapMemory(core_->Allocator(), staging_allocation);
  core_->SingleTimeCommand([&](VkCommandBuffer command_buffer) {
    VkImageAspectFlagBits aspect = IsDepthFormat(format_) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    vulkan::TransitImageLayout(command_buffer, image_, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                               VK_ACCESS_TRANSFER_WRITE_BIT, aspect);
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
    vkCmdCopyBufferToImage(command_buffer, staging_buffer, image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    vulkan::TransitImageLayout(command_buffer, image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                               VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
                               VK_ACCESS_TRANSFER_WRITE_BIT, 0, aspect);
  });
  vmaDestroyBuffer(core_->Allocator(), staging_buffer, staging_allocation);
}

void VulkanImage::DownloadData(void *data) const {
  auto extent = extent_;
  auto pixel_size = static_cast<size_t>(PixelSize(format_));
  VkBuffer staging_buffer = VK_NULL_HANDLE;
  VmaAllocation staging_allocation = VK_NULL_HANDLE;
  core_->CreateBuffer(pixel_size * extent.width * extent.height, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      VMA_MEMORY_USAGE_CPU_ONLY, &staging_buffer, &staging_allocation);
  core_->SingleTimeCommand([&](VkCommandBuffer command_buffer) {
    VkImageAspectFlagBits aspect = IsDepthFormat(format_) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    vulkan::TransitImageLayout(command_buffer, image_, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                               VK_ACCESS_TRANSFER_READ_BIT, aspect);
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
    vkCmdCopyImageToBuffer(command_buffer, image_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging_buffer, 1, &region);
    vulkan::TransitImageLayout(command_buffer, image_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                               VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
                               VK_ACCESS_TRANSFER_READ_BIT, 0, aspect);
  });

  void *mapped = nullptr;
  vmaMapMemory(core_->Allocator(), staging_allocation, &mapped);
  std::memcpy(data, mapped, pixel_size * extent.width * extent.height);
  vmaUnmapMemory(core_->Allocator(), staging_allocation);
  vmaDestroyBuffer(core_->Allocator(), staging_buffer, staging_allocation);
}

void VulkanImage::UploadData(const void *data, const Offset2D &offset, const Extent2D &extent) const {
  auto pixel_size = static_cast<size_t>(PixelSize(format_));
  VkBuffer staging_buffer = VK_NULL_HANDLE;
  VmaAllocation staging_allocation = VK_NULL_HANDLE;
  core_->CreateBuffer(pixel_size * extent.width * extent.height, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      VMA_MEMORY_USAGE_CPU_ONLY, &staging_buffer, &staging_allocation);
  void *mapped = nullptr;
  vmaMapMemory(core_->Allocator(), staging_allocation, &mapped);
  std::memcpy(mapped, data, pixel_size * extent.width * extent.height);
  vmaUnmapMemory(core_->Allocator(), staging_allocation);
  core_->SingleTimeCommand([&](VkCommandBuffer command_buffer) {
    VkImageAspectFlagBits aspect = IsDepthFormat(format_) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    vulkan::TransitImageLayout(command_buffer, image_, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                               VK_ACCESS_TRANSFER_WRITE_BIT, aspect);
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
    region.imageOffset = {offset.x, offset.y, 0};
    region.imageExtent = {extent.width, extent.height, 1};
    vkCmdCopyBufferToImage(command_buffer, staging_buffer, image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    vulkan::TransitImageLayout(command_buffer, image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                               VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
                               VK_ACCESS_TRANSFER_WRITE_BIT, 0, aspect);
  });
  vmaDestroyBuffer(core_->Allocator(), staging_buffer, staging_allocation);
}

void VulkanImage::DownloadData(void *data, const Offset2D &offset, const Extent2D &extent) const {
  auto pixel_size = static_cast<size_t>(PixelSize(format_));
  VkBuffer staging_buffer = VK_NULL_HANDLE;
  VmaAllocation staging_allocation = VK_NULL_HANDLE;
  core_->CreateBuffer(pixel_size * extent.width * extent.height, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      VMA_MEMORY_USAGE_CPU_ONLY, &staging_buffer, &staging_allocation);
  core_->SingleTimeCommand([&](VkCommandBuffer command_buffer) {
    VkImageAspectFlagBits aspect = IsDepthFormat(format_) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    vulkan::TransitImageLayout(command_buffer, image_, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                               VK_ACCESS_TRANSFER_READ_BIT, aspect);
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
    region.imageOffset = {offset.x, offset.y, 0};
    region.imageExtent = {extent.width, extent.height, 1};
    vkCmdCopyImageToBuffer(command_buffer, image_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging_buffer, 1, &region);
    vulkan::TransitImageLayout(command_buffer, image_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                               VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
                               VK_ACCESS_TRANSFER_READ_BIT, 0, aspect);
  });

  void *mapped = nullptr;
  vmaMapMemory(core_->Allocator(), staging_allocation, &mapped);
  std::memcpy(data, mapped, pixel_size * extent.width * extent.height);
  vmaUnmapMemory(core_->Allocator(), staging_allocation);
  vmaDestroyBuffer(core_->Allocator(), staging_buffer, staging_allocation);
}

}  // namespace grassland::graphics::backend
