#pragma once
#include "cao_di/vulkan/device.h"
#include "cao_di/vulkan/vulkan_util.h"

namespace CD::vulkan {
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
}  // namespace CD::vulkan
