#pragma once
#include "grassland/rhi/backend/vulkan/vulkan_sampler.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
class VulkanSamplerImpl final : public Object<IVulkanSampler> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(ISampler, IVulkanSampler, IBackendSampler, ISampler)

  static IVulkanSampler *Create(IVulkanDevice *device, const SamplerDesc &desc);

  ~VulkanSamplerImpl() override;

  VkSampler NativeSampler() const {
    return sampler_;
  }

  const SamplerDesc &GetDesc() const override {
    return desc_;
  }

  const void *DeviceIdentity() const override {
    return device_identity_;
  }

 private:
  VulkanSamplerImpl(IVulkanDevice *device, const SamplerDesc &desc);

  VkDevice device_;

  VkSampler sampler_{};

  void InitializeSampler(const SamplerDesc &desc, const void *device) {
    desc_ = desc;
    device_identity_ = device;
  }

  SamplerDesc desc_{};

  const void *device_identity_{};
};
}  // namespace grassland::rhi::backend::vulkan
