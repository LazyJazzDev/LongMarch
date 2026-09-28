#include "grassland/graphics/backend/vulkan/helper/pipeline_layout.h"

#include "grassland/graphics/backend/vulkan/vulkan_core.h"

namespace grassland::graphics::backend::vulkan {
PipelineLayout::PipelineLayout(const VulkanCore *device, VkPipelineLayout pipeline_layout)
    : device_(device),
      pipeline_layout_(pipeline_layout) {
}

PipelineLayout::~PipelineLayout() {
  vkDestroyPipelineLayout(device_->Handle(), pipeline_layout_, nullptr);
}
}  // namespace grassland::graphics::backend::vulkan
