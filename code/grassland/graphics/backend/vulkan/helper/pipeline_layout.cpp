#include "grassland/graphics/backend/vulkan/helper/pipeline_layout.h"

namespace grassland::graphics::backend::vulkan {
PipelineLayout::PipelineLayout(const struct Device *device, VkPipelineLayout pipeline_layout)
    : device_(device),
      pipeline_layout_(pipeline_layout) {
}

PipelineLayout::~PipelineLayout() {
  vkDestroyPipelineLayout(device_->Handle(), pipeline_layout_, nullptr);
}
}  // namespace grassland::graphics::backend::vulkan
