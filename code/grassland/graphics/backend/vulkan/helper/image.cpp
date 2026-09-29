#include "grassland/graphics/backend/vulkan/helper/image.h"

namespace grassland::graphics::backend::vulkan {
void TransitImageLayout(VkCommandBuffer command_buffer,
                        VkImage image,
                        VkImageLayout old_layout,
                        VkImageLayout new_layout,
                        VkPipelineStageFlags src_stage_flags,
                        VkPipelineStageFlags dst_stage_flags,
                        VkAccessFlags src_access_flags,
                        VkAccessFlags dst_access_flags,
                        VkImageAspectFlags aspect) {
  VkImageMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = old_layout;
  barrier.newLayout = new_layout;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image;
  barrier.subresourceRange.aspectMask = aspect;
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;
  barrier.srcAccessMask = src_access_flags;
  barrier.dstAccessMask = dst_access_flags;

  vkCmdPipelineBarrier(command_buffer, src_stage_flags, dst_stage_flags, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

}  // namespace grassland::graphics::backend::vulkan
