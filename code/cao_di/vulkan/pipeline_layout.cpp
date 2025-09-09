#include "cao_di/vulkan/pipeline_layout.h"

namespace CD::vulkan {
PipelineLayout::PipelineLayout(const struct Device *device, VkPipelineLayout pipeline_layout)
    : device_(device), pipeline_layout_(pipeline_layout) {
}

PipelineLayout::~PipelineLayout() {
  vkDestroyPipelineLayout(device_->Handle(), pipeline_layout_, nullptr);
}
}  // namespace CD::vulkan
