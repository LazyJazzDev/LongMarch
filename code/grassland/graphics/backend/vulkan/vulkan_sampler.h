#pragma once
#include "grassland/graphics/backend/vulkan/vulkan_core.h"
#include "grassland/graphics/backend/vulkan/vulkan_util.h"

namespace grassland::graphics::backend {

class VulkanSampler : public Sampler {
 public:
  VulkanSampler(VulkanCore *core, const SamplerInfo &info);
  vulkan::Sampler *Sampler() const {
    return sampler_.get();
  }

 private:
  VulkanCore *core_;
  std::unique_ptr<vulkan::Sampler> sampler_;
};

}  // namespace grassland::graphics::backend
