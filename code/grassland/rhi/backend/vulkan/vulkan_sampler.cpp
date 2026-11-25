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
  if (!ValidateSamplerFeatures(desc, device->GetPhysicalDevice().mip_feature))
    return nullptr;
  auto sampler = ObjectPtr<VulkanSamplerImpl>::Adopt(new VulkanSamplerImpl(device, desc));
  const VkSamplerAddressMode modes[] = {VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                        VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
                                        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER};
  VkSamplerCreateInfo create{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  create.minFilter = desc.MinFilter() == SAMPLER_FILTER_LINEAR ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
  create.magFilter = desc.MagFilter() == SAMPLER_FILTER_LINEAR ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
  create.mipmapMode =
      desc.MipFilter() == SAMPLER_FILTER_LINEAR ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
  create.addressModeU = modes[desc.address_u];
  create.addressModeV = modes[desc.address_v];
  create.addressModeW = modes[desc.address_w];
  const VkBorderColor colors[] = {VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK, VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,
                                  VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE};
  create.borderColor = colors[desc.border_color];
  create.compareEnable = desc.compare_enable;
  create.compareOp = static_cast<VkCompareOp>(desc.compare_op);
  create.minLod = desc.min_lod;
  create.maxLod = desc.max_lod;
  create.mipLodBias = desc.mip_lod_bias;
  create.anisotropyEnable = desc.max_anisotropy > 1;
  create.maxAnisotropy = static_cast<float>(desc.max_anisotropy);
  if (!Check(vkCreateSampler(device->NativeDevice(), &create, nullptr, &sampler->sampler_), "Create sampler"))
    return nullptr;
  return sampler.Detach();
}

rhi::ISampler *IVulkanSampler::Create(IVulkanDevice *device, const SamplerDesc &desc) {
  return VulkanSamplerImpl::Create(device, desc);
}
}  // namespace grassland::rhi::backend::vulkan
