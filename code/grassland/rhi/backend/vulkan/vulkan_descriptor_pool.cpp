#include "grassland/rhi/backend/vulkan/vulkan_descriptor_pool.h"

#include <algorithm>
#include <limits>

namespace grassland::rhi::backend::vulkan {
VulkanDescriptorPool::Lease::Lease(VulkanDescriptorPool *owner, Page *page, VkDescriptorSet set)
    : owner_(owner),
      page_(page),
      set_(set) {
}

VulkanDescriptorPool::Lease::~Lease() {
  owner_->ReleaseSet(page_, set_);
}

VkDescriptorSet VulkanDescriptorPool::Lease::Set() const {
  return set_;
}

VulkanDescriptorPool::VulkanDescriptorPool(VkDevice device) : device_(device) {
}

size_t VulkanDescriptorPool::PatternHash::operator()(const Pattern &pattern) const {
  size_t hash = 0;
  for (auto count : pattern)
    hash ^= std::hash<uint32_t>{}(count) + size_t(0x9e3779b9) + (hash << 6) + (hash >> 2);
  return hash;
}

VulkanDescriptorPool::~VulkanDescriptorPool() {
  for (auto &[pattern, pages] : pages_)
    for (auto &page : pages)
      vkDestroyDescriptorPool(device_, page->pool, nullptr);
}

ObjectPtr<VulkanDescriptorPool::Lease> VulkanDescriptorPool::Acquire(VkDescriptorSetLayout layout,
                                                                     const BindingLayoutDesc &desc) {
  try {
    Pattern pattern{};
    for (auto binding : desc.bindings) {
      auto &count = pattern[DescriptorTypeIndex(binding.type)];
      if (binding.count > std::numeric_limits<uint32_t>::max() - count)
        return {};
      count += binding.count;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto &pages = pages_[pattern];
    Page *page = nullptr;
    for (auto &candidate : pages) {
      if (candidate->available_sets) {
        page = candidate.get();
        break;
      }
    }
    if (!page) {
      auto fresh = std::make_unique<Page>();
      fresh->pattern = pattern;
      fresh->capacity = 64;
      for (auto count : pattern)
        if (count)
          fresh->capacity = std::min(fresh->capacity, std::numeric_limits<uint32_t>::max() / count);
      fresh->available_sets = fresh->capacity;
      std::vector<VkDescriptorPoolSize> sizes;
      for (size_t i = 0; i < pattern.size(); ++i)
        if (pattern[i])
          sizes.push_back({DescriptorTypeFromIndex(i), pattern[i] * fresh->capacity});
      VkDescriptorPoolCreateInfo create{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
      create.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
      create.maxSets = fresh->capacity;
      create.poolSizeCount = uint32_t(sizes.size());
      create.pPoolSizes = sizes.data();
      pages.reserve(pages.size() + 1);
      if (!Check(vkCreateDescriptorPool(device_, &create, nullptr, &fresh->pool), "Create descriptor page"))
        return {};
      page = fresh.get();
      pages.push_back(std::move(fresh));
    }
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate.descriptorPool = page->pool;
    allocate.descriptorSetCount = 1;
    allocate.pSetLayouts = &layout;
    VkDescriptorSet set{};
    if (!Check(vkAllocateDescriptorSets(device_, &allocate, &set), "Allocate descriptor set"))
      return {};
    --page->available_sets;
    try {
      return ObjectPtr<Lease>::Adopt(new Lease(this, page, set));
    } catch (const std::bad_alloc &) {
      if (Check(vkFreeDescriptorSets(device_, page->pool, 1, &set), "Free descriptor set"))
        ++page->available_sets;
      return {};
    }
  } catch (const std::bad_alloc &) {
    return {};
  }
}

void VulkanDescriptorPool::ReleaseSet(Page *page, VkDescriptorSet set) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!Check(vkFreeDescriptorSets(device_, page->pool, 1, &set), "Free descriptor set"))
    return;
  ++page->available_sets;
  if (page->available_sets != page->capacity)
    return;
  Check(vkResetDescriptorPool(device_, page->pool, 0), "Reset descriptor page");
  auto &pages = pages_.at(page->pattern);
  // Retain one idle page per pattern, rather than across unrelated layouts.
  if (std::any_of(pages.begin(), pages.end(),
                  [page](const auto &p) { return p.get() != page && p->available_sets == p->capacity; })) {
    vkDestroyDescriptorPool(device_, page->pool, nullptr);
    pages.erase(std::find_if(pages.begin(), pages.end(), [page](const auto &p) { return p.get() == page; }));
  }
}

std::pair<size_t, size_t> VulkanDescriptorPool::GetPageCounts() const {
  std::lock_guard<std::mutex> lock(mutex_);
  size_t total = 0, occupied = 0;
  for (const auto &[pattern, pages] : pages_) {
    total += pages.size();
    occupied +=
        std::count_if(pages.begin(), pages.end(), [](const auto &p) { return p->available_sets != p->capacity; });
  }
  return {total, occupied};
}

}  // namespace grassland::rhi::backend::vulkan
