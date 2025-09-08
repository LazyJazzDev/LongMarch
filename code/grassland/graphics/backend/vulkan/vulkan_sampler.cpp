#include "grassland/graphics/backend/vulkan/vulkan_sampler.h"

namespace CD::graphics::backend {

VulkanSampler::VulkanSampler(VulkanCore *core, const SamplerInfo &info) : core_(core) {
  VkBorderColor border_color = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
  core_->Device()->CreateSampler(
      FilterModeToVkFilter(info.min_filter), FilterModeToVkFilter(info.mag_filter),
      AddressModeToVkSamplerAddressMode(info.address_mode_u), AddressModeToVkSamplerAddressMode(info.address_mode_v),
      AddressModeToVkSamplerAddressMode(info.address_mode_w), VK_FALSE, VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
      FilterModeToVkSamplerMipmapMode(info.mip_filter), &sampler_);
}

}  // namespace CD::graphics::backend
