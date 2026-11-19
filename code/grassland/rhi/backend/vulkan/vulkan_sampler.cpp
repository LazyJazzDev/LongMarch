#include "grassland/rhi/backend/vulkan/vulkan_sampler.h"

#include "grassland/rhi/backend/vulkan/vulkan_device.h"

namespace grassland::rhi::backend::vulkan {
VulkanSamplerImpl::VulkanSamplerImpl(IVulkanDevice *device, const SamplerDesc &desc) : device_(device->NativeDevice()) {
  InitializeSampler(desc, device);
}

VulkanSamplerImpl::~VulkanSamplerImpl() {
  if (sampler_)
    vkDestroySampler(device_, sampler_, nullptr);
}

rhi::ISampler *VulkanSamplerImpl::Create(IVulkanDevice *device, const SamplerDesc &desc) {
  if (!ValidateSamplerDesc(desc))
    return nullptr;
  VkPhysicalDeviceFeatures features{};
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceFeatures(device->NativePhysicalDevice(), &features);
  vkGetPhysicalDeviceProperties(device->NativePhysicalDevice(), &properties);
  if (desc.max_anisotropy > 1 &&
      (!features.samplerAnisotropy || desc.max_anisotropy > properties.limits.maxSamplerAnisotropy))
    return nullptr;
  auto sampler = ObjectPtr<VulkanSamplerImpl>::Adopt(new VulkanSamplerImpl(device, desc));
  const VkSamplerAddressMode modes[] = {VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                        VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT};
  VkSamplerCreateInfo create{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  create.magFilter = create.minFilter = desc.filter == SAMPLER_FILTER_LINEAR ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
  create.mipmapMode =
      desc.mip_filter == SAMPLER_FILTER_LINEAR ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
  create.anisotropyEnable = desc.max_anisotropy > 1;
  create.maxAnisotropy = float(desc.max_anisotropy);
  create.compareEnable = desc.comparison.has_value();
  create.compareOp = static_cast<VkCompareOp>(desc.comparison.value_or(COMPARE_ALWAYS));
  create.addressModeU = modes[desc.address_u];
  create.addressModeV = create.addressModeW = modes[desc.address_v];
  create.minLod = desc.min_lod;
  create.maxLod = desc.max_lod;
  if (!Check(vkCreateSampler(device->NativeDevice(), &create, nullptr, &sampler->sampler_), "Create sampler"))
    return nullptr;
  return sampler.Detach();
}

rhi::ISampler *IVulkanSampler::Create(IVulkanDevice *device, const SamplerDesc &desc) {
  return VulkanSamplerImpl::Create(device, desc);
}
}  // namespace grassland::rhi::backend::vulkan
