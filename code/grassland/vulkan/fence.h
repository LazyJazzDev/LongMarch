#pragma once
#include "grassland/vulkan/device.h"
#include "grassland/vulkan/vulkan_utils.h"

namespace grassland::vulkan {
class Fence {
 public:
  Fence(const class Device *device, VkFence fence);

  ~Fence();

  [[nodiscard]] VkFence Handle() const {
    return fence_;
  }

  [[nodiscard]] const class Device *Device() const {
    return device_;
  }

 private:
  const class Device *device_{};
  VkFence fence_{};
};
}  // namespace grassland::vulkan
