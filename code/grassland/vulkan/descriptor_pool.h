#pragma once

#include "grassland/vulkan/device.h"

namespace grassland::vulkan {

struct DescriptorPoolSize {
  std::map<VkDescriptorType, uint32_t> descriptor_type_count;

  explicit DescriptorPoolSize(uint32_t count = 0) {
    descriptor_type_count[VK_DESCRIPTOR_TYPE_SAMPLER] = count;
    descriptor_type_count[VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER] = count;
    descriptor_type_count[VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE] = count;
    descriptor_type_count[VK_DESCRIPTOR_TYPE_STORAGE_IMAGE] = count;
    descriptor_type_count[VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER] = count;
    descriptor_type_count[VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER] = count;
    descriptor_type_count[VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER] = count;
    descriptor_type_count[VK_DESCRIPTOR_TYPE_STORAGE_BUFFER] = count;
    descriptor_type_count[VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC] = count;
    descriptor_type_count[VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC] = count;
    descriptor_type_count[VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT] = count;
  }

  DescriptorPoolSize(VkDescriptorType type, uint32_t count) {
    descriptor_type_count[type] = count;
  }

  DescriptorPoolSize operator+(const DescriptorPoolSize &other) const {
    DescriptorPoolSize result = *this;
    for (auto &[type, count] : other.descriptor_type_count) {
      result.descriptor_type_count[type] += count;
    }
    return result;
  }

  DescriptorPoolSize &operator+=(const DescriptorPoolSize &other) {
    for (auto &[type, count] : other.descriptor_type_count) {
      descriptor_type_count[type] += count;
    }
    return *this;
  }

  DescriptorPoolSize operator*(uint32_t multiplier) const {
    DescriptorPoolSize result;
    for (auto &[type, count] : descriptor_type_count) {
      result.descriptor_type_count[type] = count * multiplier;
    }
    return result;
  }

  DescriptorPoolSize operator*=(uint32_t multiplier) {
    for (auto &[type, count] : descriptor_type_count) {
      count *= multiplier;
    }
    return *this;
  }

  friend DescriptorPoolSize operator*(uint32_t multiplier,
                                      const DescriptorPoolSize &pool_size) {
    return pool_size * multiplier;
  }

  std::vector<VkDescriptorPoolSize> ToVkDescriptorPoolSize() const {
    return *this;
  }

  operator std::vector<VkDescriptorPoolSize>() const {
    std::vector<VkDescriptorPoolSize> pool_sizes;
    pool_sizes.reserve(descriptor_type_count.size());
    for (auto &[type, count] : descriptor_type_count) {
      if (count) {
        pool_sizes.push_back({type, count});
      }
    }
    return pool_sizes;
  }
};

class DescriptorPool {
 public:
  DescriptorPool(const class Device *device,
                 VkDescriptorPool descriptor_pool,
                 DescriptorPoolSize pool_size,
                 uint32_t max_sets);

  ~DescriptorPool();

  const class Device *Device() const {
    return device_;
  }

  VkDescriptorPool Handle() const {
    return descriptor_pool_;
  }

  VkResult AllocateDescriptorSet(
      VkDescriptorSetLayout layout,
      double_ptr<DescriptorSet> pp_descriptor_set) const;

  const DescriptorPoolSize &PoolSize() const {
    return pool_size_;
  }

  uint32_t MaxSets() const {
    return max_sets_;
  }

 private:
  const class Device *device_{};
  VkDescriptorPool descriptor_pool_{};
  DescriptorPoolSize pool_size_;
  uint32_t max_sets_{};
};
}  // namespace grassland::vulkan
