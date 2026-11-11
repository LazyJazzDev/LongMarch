#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend::vulkan {
class IVulkanSampler;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::vulkan::IVulkanSampler)
#include "grassland/rhi/backend/backend_sampler.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
class IVulkanDevice;

class IVulkanSampler : public Sampler {
 public:
  static IVulkanSampler *Create(IVulkanDevice *device, const SamplerDesc &desc);
  virtual VkSampler NativeSampler() const = 0;
};

;
}  // namespace grassland::rhi::backend::vulkan
