#pragma once
#include "grassland/graphics/backend/vulkan/helper/native_types.h"

namespace grassland::graphics::backend::vulkan {
class PipelineLayout {
 public:
  PipelineLayout(const VulkanCore *device, VkPipelineLayout pipeline_layout);

  ~PipelineLayout();

  const VulkanCore *Device() const {
    return device_;
  }

  VkPipelineLayout Handle() const {
    return pipeline_layout_;
  }

 private:
  const VulkanCore *device_{};
  VkPipelineLayout pipeline_layout_{};
};
}  // namespace grassland::graphics::backend::vulkan
