#pragma once
#include "grassland/rhi/backend/backend_sampler.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

LONGMARCH_RHI_UUID(grassland::rhi::backend::vulkan::IVulkanSampler, "cbdfdbcb-a546-480b-a5d0-a86e1d181b71")

namespace grassland::rhi::backend::vulkan {
class IVulkanSampler : public IBackendSampler {
 public:
  static rhi::ISampler *Create(IVulkanDevice *device, const SamplerDesc &desc);
  virtual VkSampler NativeSampler() const = 0;
};

}  // namespace grassland::rhi::backend::vulkan
