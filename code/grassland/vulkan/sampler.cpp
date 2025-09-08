#include "grassland/vulkan/sampler.h"

namespace CD::vulkan {
Sampler::Sampler(const struct Device *device, VkSampler sampler) : device_(device), sampler_(sampler) {
}

Sampler::~Sampler() {
  vkDestroySampler(device_->Handle(), sampler_, nullptr);
}
}  // namespace CD::vulkan
