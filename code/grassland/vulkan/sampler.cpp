#include "grassland/vulkan/sampler.h"

namespace grassland::vulkan {
Sampler::Sampler(const struct Device *device, VkSampler sampler) : device_(device), sampler_(sampler) {
}

Sampler::~Sampler() {
  vkDestroySampler(device_->Handle(), sampler_, nullptr);
}
}  // namespace grassland::vulkan
