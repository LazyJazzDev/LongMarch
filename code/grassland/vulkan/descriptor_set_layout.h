#pragma once
#include "grassland/vulkan/descriptor_pool.h"
#include "grassland/vulkan/device.h"

namespace CD::vulkan {
class DescriptorSetLayout {
 public:
  DescriptorSetLayout(const class Device *device,
                      VkDescriptorSetLayout layout,
                      const std::vector<VkDescriptorSetLayoutBinding> &bindings);

  ~DescriptorSetLayout();

  DescriptorPoolSize GetPoolSize() const;

  VkDescriptorSetLayout Handle() const {
    return layout_;
  }

  const class Device *Device() const {
    return device_;
  }

  const std::vector<VkDescriptorSetLayoutBinding> &Bindings() const {
    return bindings_;
  }

 private:
  const class Device *device_{};

  VkDescriptorSetLayout layout_{};

  std::vector<VkDescriptorSetLayoutBinding> bindings_{};
};
}  // namespace CD::vulkan
