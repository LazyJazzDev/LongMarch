#pragma once
#include "grassland/graphics/backend/vulkan/helper/device.h"
#include "grassland/graphics/backend/vulkan/helper/vulkan_util.h"

namespace grassland::graphics::backend::vulkan {
class Fence {
 public:
  Fence(const class Device *device, VkFence fence);

  ~Fence();

  VkFence Handle() const {
    return fence_;
  }

  const class Device *Device() const {
    return device_;
  }

 private:
  const class Device *device_{};
  VkFence fence_{};
};
}  // namespace grassland::graphics::backend::vulkan
