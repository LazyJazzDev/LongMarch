#pragma once

#include <vulkan/vulkan.h>

namespace longmarch::harmony {
// XEngine extensions are enumerated separately from standard Vulkan extensions.
void LogXEngineCapabilities(VkPhysicalDevice physical_device);
}  // namespace longmarch::harmony
