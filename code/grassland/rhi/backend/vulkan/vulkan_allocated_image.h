#pragma once
#include "grassland/rhi/backend/vulkan/vulkan_image.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"
typedef struct VmaAllocation_T *VmaAllocation;

namespace grassland::rhi::backend::vulkan {
IImage *CreateVulkanAllocatedImage(IVulkanDevice *device, const ImageDesc &desc);

}  // namespace grassland::rhi::backend::vulkan
