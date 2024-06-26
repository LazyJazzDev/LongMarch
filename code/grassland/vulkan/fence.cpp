#include "grassland/vulkan/fence.h"

namespace grassland::vulkan {
Fence::Fence(const struct Device *device, VkFence fence)
    : device_(device), fence_(fence) {
}

Fence::~Fence() {
  vkDestroyFence(device_->Handle(), fence_, nullptr);
}
}  // namespace grassland::vulkan
