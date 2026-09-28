#include "grassland/graphics/backend/vulkan/vulkan_sampler.h"

namespace grassland::graphics::backend {

VulkanSampler::VulkanSampler(VulkanCore *core, const SamplerInfo &info) : core_(core) {
  VkSamplerCreateInfo create_info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  create_info.magFilter = FilterModeToVkFilter(info.mag_filter);
  create_info.minFilter = FilterModeToVkFilter(info.min_filter);
  create_info.addressModeU = AddressModeToVkSamplerAddressMode(info.address_mode_u);
  create_info.addressModeV = AddressModeToVkSamplerAddressMode(info.address_mode_v);
  create_info.addressModeW = AddressModeToVkSamplerAddressMode(info.address_mode_w);
  create_info.maxAnisotropy = 1.0f;
  create_info.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
  create_info.compareOp = VK_COMPARE_OP_ALWAYS;
  create_info.mipmapMode = FilterModeToVkSamplerMipmapMode(info.mip_filter);
  vulkan::ThrowIfFailed(vkCreateSampler(core_->Device()->Handle(), &create_info, nullptr, &sampler_),
                        "Failed to create Vulkan sampler");
}

VulkanSampler::~VulkanSampler() {
  vkDestroySampler(core_->Device()->Handle(), sampler_, nullptr);
}

}  // namespace grassland::graphics::backend
