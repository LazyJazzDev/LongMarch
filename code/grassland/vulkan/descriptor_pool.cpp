#include "grassland/vulkan/descriptor_pool.h"

#include <utility>

#include "grassland/vulkan/descriptor_set.h"

namespace grassland::vulkan {
DescriptorPool::DescriptorPool(const struct Device *device,
                               VkDescriptorPool descriptor_pool,
                               DescriptorPoolSize pool_size,
                               uint32_t max_sets)
    : device_(device),
      descriptor_pool_(descriptor_pool),
      pool_size_(std::move(pool_size)),
      max_sets_(max_sets) {
}

DescriptorPool::~DescriptorPool() {
  vkDestroyDescriptorPool(device_->Handle(), descriptor_pool_, nullptr);
}

VkResult DescriptorPool::AllocateDescriptorSet(
    VkDescriptorSetLayout layout,
    double_ptr<DescriptorSet> pp_descriptor_set) const {
  if (!pp_descriptor_set) {
    SetErrorMessage("pp_descriptor_sets is nullptr");
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  VkDescriptorSetAllocateInfo allocate_info{};
  allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocate_info.descriptorPool = descriptor_pool_;
  allocate_info.descriptorSetCount = 1;
  allocate_info.pSetLayouts = &layout;

  VkDescriptorSet descriptor_set;
  RETURN_IF_FAILED_VK(vkAllocateDescriptorSets(device_->Handle(),
                                               &allocate_info, &descriptor_set),
                      "failed to allocate descriptor set");

  pp_descriptor_set.construct(this, descriptor_set);

  return VK_SUCCESS;
}
}  // namespace grassland::vulkan
