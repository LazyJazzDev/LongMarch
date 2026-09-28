#pragma once
#include "grassland/graphics/backend/vulkan/helper/descriptor_pool.h"
#include "grassland/graphics/backend/vulkan/helper/native_types.h"

namespace grassland::graphics::backend::vulkan {
class DescriptorSetLayout {
 public:
  DescriptorSetLayout(const VulkanCore *device,
                      VkDescriptorSetLayout layout,
                      const std::vector<VkDescriptorSetLayoutBinding> &bindings);

  ~DescriptorSetLayout();

  DescriptorPoolSize GetPoolSize() const;

  VkDescriptorSetLayout Handle() const {
    return layout_;
  }

  const VulkanCore *Device() const {
    return device_;
  }

  const std::vector<VkDescriptorSetLayoutBinding> &Bindings() const {
    return bindings_;
  }

 private:
  const VulkanCore *device_{};

  VkDescriptorSetLayout layout_{};

  std::vector<VkDescriptorSetLayoutBinding> bindings_{};
};
}  // namespace grassland::graphics::backend::vulkan
