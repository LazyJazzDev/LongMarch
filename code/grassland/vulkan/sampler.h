#pragma once
#include "grassland/vulkan/device.h"

namespace grassland::vulkan {
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
}  // namespace grassland::vulkan
