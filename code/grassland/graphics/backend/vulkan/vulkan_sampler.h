#pragma once
#include "grassland/graphics/backend/vulkan/vulkan_core.h"
#include "grassland/graphics/backend/vulkan/vulkan_util.h"

namespace grassland::graphics::backend {

class VulkanSampler : public Sampler {
 public:
  VulkanSampler(VulkanCore *core, const SamplerInfo &info);
  ~VulkanSampler() override;

  VkSampler Handle() const {
    return sampler_;
  }

 private:
  VulkanCore *core_;
  VkSampler sampler_{VK_NULL_HANDLE};
};

}  // namespace grassland::graphics::backend
