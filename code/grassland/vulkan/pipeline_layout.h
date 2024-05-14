#pragma once
#include "grassland/vulkan/device.h"

namespace grassland::vulkan {
class PipelineLayout {
 public:
  PipelineLayout(const class Device *device, VkPipelineLayout pipeline_layout);

  ~PipelineLayout();

  const class Device *Device() const {
    return device_;
  }

  VkPipelineLayout Handle() const {
    return pipeline_layout_;
  }

 private:
  const class Device *device_{};
  VkPipelineLayout pipeline_layout_{};
};
}  // namespace grassland::vulkan
