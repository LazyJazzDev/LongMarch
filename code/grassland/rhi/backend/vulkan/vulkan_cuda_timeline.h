#pragma once
#include "grassland/rhi/backend/vulkan/vulkan_timeline.h"
#include "grassland/rhi/rhi_cuda_timeline.h"
#if defined(LONGMARCH_CUDA_RUNTIME)
namespace grassland::rhi::backend::vulkan {
ObjectPtr<IVulkanTimeline> CreateVulkanCUDATimeline(IVulkanDevice *device);
}
#endif
