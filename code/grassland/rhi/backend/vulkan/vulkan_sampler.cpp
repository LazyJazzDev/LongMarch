#include "grassland/rhi/backend/vulkan/vulkan_sampler.h"

#include "grassland/rhi/backend/vulkan/vulkan_device.h"
#include "grassland/rhi/backend/vulkan/vulkan_sampler_impl.h"

namespace grassland::rhi::backend::vulkan {
VulkanSamplerImpl::VulkanSamplerImpl(IVulkanDevice *device, const SamplerDesc &desc) : device_(device->NativeDevice()) {
  InitializeSampler(desc, device);
}

VulkanSamplerImpl::~VulkanSamplerImpl() {
  if (sampler_)
    vkDestroySampler(device_, sampler_, nullptr);
}

IVulkanSampler *VulkanSamplerImpl::Create(IVulkanDevice *device, const SamplerDesc &desc) {
  if (!Validate(desc))
    return nullptr;
  auto sampler = ObjectPtr<VulkanSamplerImpl>::Adopt(new VulkanSamplerImpl(device, desc));
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
}

IVulkanSampler *IVulkanSampler::Create(IVulkanDevice *device, const SamplerDesc &desc) {
  return VulkanSamplerImpl::Create(device, desc);
}
}  // namespace grassland::rhi::backend::vulkan

LONGMARCH_RHI_UUID(grassland::rhi::backend::vulkan::IVulkanSampler, "cbdfdbcb-a546-480b-a5d0-a86e1d181b71")
