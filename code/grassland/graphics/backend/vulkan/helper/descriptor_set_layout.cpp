#include "grassland/graphics/backend/vulkan/helper/descriptor_set_layout.h"

#include "grassland/graphics/backend/vulkan/vulkan_core.h"

namespace grassland::graphics::backend::vulkan {

DescriptorSetLayout::DescriptorSetLayout(const VulkanCore *device,
                                         VkDescriptorSetLayout layout,
                                         const std::vector<VkDescriptorSetLayoutBinding> &bindings)
    : device_(device),
      layout_(layout),
      bindings_(bindings) {
}

DescriptorSetLayout::~DescriptorSetLayout() {
  vkDestroyDescriptorSetLayout(device_->Handle(), layout_, nullptr);
}

DescriptorPoolSize DescriptorSetLayout::GetPoolSize() const {
  DescriptorPoolSize pool_size{};
  for (const auto &binding : bindings_) {
    pool_size.descriptor_type_count[binding.descriptorType] += binding.descriptorCount;
  }
  return pool_size;
}
}  // namespace grassland::graphics::backend::vulkan
