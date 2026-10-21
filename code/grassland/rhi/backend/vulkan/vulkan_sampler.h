#pragma once
#include "grassland/rhi/backend/backend_sampler.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
class VulkanDevice;

class VulkanSampler final : public Sampler {
 public:
  static VulkanSampler *Create(VulkanDevice *device, const SamplerDesc &desc);
  ~VulkanSampler() override;

  VkSampler NativeSampler() const {
    return sampler_;
  }

 private:
  VulkanSampler(VulkanDevice *device, const SamplerDesc &desc);
  VkDevice device_;
  VkSampler sampler_{};
};
}  // namespace grassland::rhi::backend::vulkan
