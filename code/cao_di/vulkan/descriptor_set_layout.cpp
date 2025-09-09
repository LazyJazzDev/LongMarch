#include "cao_di/vulkan/descriptor_set_layout.h"

namespace CD::vulkan {

DescriptorSetLayout::DescriptorSetLayout(const struct Device *device,
                                         VkDescriptorSetLayout layout,
                                         const std::vector<VkDescriptorSetLayoutBinding> &bindings)
    : device_(device), layout_(layout), bindings_(bindings) {
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
}  // namespace CD::vulkan
