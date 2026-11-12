#pragma once

#include "grassland/rhi/rhi.h"
#ifdef LONGMARCH_VULKAN_ENABLED
#include "grassland/rhi/backend/vulkan/vulkan_device.h"
#endif

namespace grassland::rhi {
// D24S8 is optional on Vulkan. Check native capabilities before creating it;
// a failed RHI allocation on a supported device must still fail the test.
inline bool SupportsD24S8(IDevice *device, BackendAPI api) {
#ifdef LONGMARCH_VULKAN_ENABLED
  if (api == BACKEND_API_VULKAN) {
    auto native = rhi::QueryInterfacePtr<backend::vulkan::IVulkanDevice>(device).Get();
    VkFormatProperties properties{};
    vkGetPhysicalDeviceFormatProperties(native->NativePhysicalDevice(), VK_FORMAT_D24_UNORM_S8_UINT, &properties);
    constexpr auto required = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                              VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    return (properties.optimalTilingFeatures & required) == required;
  }
#endif
  return true;
}
}  // namespace grassland::rhi
