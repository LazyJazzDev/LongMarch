#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend::vulkan {
class VulkanSampler;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::vulkan::VulkanSampler)
#include "grassland/rhi/backend/backend_sampler.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
class VulkanDevice;

class VulkanSampler final : public Sampler {
 public:
  LONGMARCH_RHI_QUERY(VulkanSampler, Sampler)

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
