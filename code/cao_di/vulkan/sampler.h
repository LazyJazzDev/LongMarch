#pragma once
#include "cao_di/vulkan/device.h"

namespace CD::vulkan {
class Sampler {
 public:
  Sampler(const class Device *device, VkSampler sampler);

  ~Sampler();

  VkSampler Handle() const {
    return sampler_;
  }

  const class Device *Device() const {
    return device_;
  }

 private:
  const class Device *device_;
  VkSampler sampler_{};
};
}  // namespace CD::vulkan
