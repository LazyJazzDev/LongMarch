#pragma once

#include "grassland/graphics/backend/vulkan/helper/native_types.h"

namespace grassland::graphics::backend::vulkan {
void TransitImageLayout(VkCommandBuffer command_buffer,
                        VkImage image,
                        VkImageLayout old_layout,
                        VkImageLayout new_layout,
                        VkPipelineStageFlags src_stage_flags,
                        VkPipelineStageFlags dst_stage_flags,
                        VkAccessFlags src_access_flags,
                        VkAccessFlags dst_access_flags,
                        VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT);

}  // namespace grassland::graphics::backend::vulkan
