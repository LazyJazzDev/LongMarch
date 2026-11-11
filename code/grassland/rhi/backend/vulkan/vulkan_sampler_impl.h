#pragma once
#include "grassland/rhi/backend/backend_sampler_state.h"
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/backend/vulkan/vulkan_sampler.h"

namespace grassland::rhi::backend::vulkan {
class VulkanSamplerImpl final : public Object<IVulkanSampler> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(ISampler, IVulkanSampler, IBackendSampler, ISampler)
  static IVulkanSampler *Create(IVulkanDevice *device, const SamplerDesc &desc);
  ~VulkanSamplerImpl() override;

  VkSampler NativeSampler() const {
    return sampler_;
  }

 private:
  VulkanSamplerImpl(IVulkanDevice *device, const SamplerDesc &desc);
  VkDevice device_;
  VkSampler sampler_{};

 private:
  SamplerState state_Sampler_;

  SamplerState &StateSampler() override {
    return state_Sampler_;
  }

  const SamplerState &StateSampler() const override {
    return state_Sampler_;
  }

 public:
  const SamplerDesc &GetDesc() const override {
    return IBackendSampler::CommonGetDesc();
  }
};
}  // namespace grassland::rhi::backend::vulkan
