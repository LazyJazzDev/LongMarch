#pragma once
#include "grassland/rhi/backend/backend_sampler.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::vulkan::IVulkanSampler,
                        grassland::rhi::IObject,
                        "cbdfdbcb-a546-480b-a5d0-a86e1d181b71")

namespace grassland::rhi::backend::vulkan {
class IVulkanSampler : public IObject {
 public:
  virtual const void *DeviceIdentity() const = 0;
  static rhi::ISampler *Create(IVulkanDevice *device, const SamplerDesc &desc);
  virtual VkSampler NativeSampler() const = 0;
};

}  // namespace grassland::rhi::backend::vulkan
