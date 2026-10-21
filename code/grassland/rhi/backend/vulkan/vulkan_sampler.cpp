#include "grassland/rhi/backend/vulkan/vulkan_sampler.h"

#include "grassland/rhi/backend/vulkan/vulkan_device.h"

namespace grassland::rhi::backend::vulkan {
VulkanSampler::VulkanSampler(VulkanDevice *device, const SamplerDesc &desc)
    : Sampler(desc, device),
      device_(device->NativeDevice()) {
}

VulkanSampler::~VulkanSampler() {
  if (sampler_)
    vkDestroySampler(device_, sampler_, nullptr);
}

VulkanSampler *VulkanSampler::Create(VulkanDevice *device, const SamplerDesc &desc) {
  if (!Validate(desc))
    return nullptr;
  try {
    auto sampler = ObjectPtr<VulkanSampler>::Adopt(new VulkanSampler(device, desc));
    const VkSamplerAddressMode modes[] = {VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                          VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT};
    VkSamplerCreateInfo create{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    create.magFilter = create.minFilter = desc.filter == SAMPLER_FILTER_LINEAR ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    create.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    create.addressModeU = modes[desc.address_u];
    create.addressModeV = create.addressModeW = modes[desc.address_v];
    create.maxLod = VK_LOD_CLAMP_NONE;
    if (!Check(vkCreateSampler(device->NativeDevice(), &create, nullptr, &sampler->sampler_), "Create sampler"))
      return nullptr;
    return sampler.Detach();
  } catch (const std::bad_alloc &) {
    return nullptr;
  }
}
}  // namespace grassland::rhi::backend::vulkan
