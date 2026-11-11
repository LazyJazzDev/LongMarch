#pragma once
#include "grassland/rhi/backend/backend_sampler.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::vulkan::IVulkanSampler)

namespace grassland::rhi::backend::vulkan {
class IVulkanSampler : public IBackendSampler {
 public:
  static IVulkanSampler *Create(IVulkanDevice *device, const SamplerDesc &desc);
  virtual VkSampler NativeSampler() const = 0;
};

}  // namespace grassland::rhi::backend::vulkan
