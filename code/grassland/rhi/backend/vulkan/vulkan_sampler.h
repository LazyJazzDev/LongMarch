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

namespace grassland::rhi::backend::vulkan {
class VulkanSamplerImpl final : public Object<rhi::ISampler, IVulkanSampler> {
 public:
  static rhi::ISampler *Create(IVulkanDevice *device, const SamplerDesc &desc);

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
