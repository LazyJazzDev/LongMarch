#pragma once

#include "cao_di/vulkan/descriptor_pool.h"
#include "cao_di/vulkan/descriptor_set_layout.h"

namespace CD::vulkan {
class DescriptorSet {
 public:
  DescriptorSet(const class DescriptorPool *descriptor_pool, VkDescriptorSet set);

  ~DescriptorSet();

  VkDescriptorSet Handle() const {
    return set_;
  }

  const class DescriptorPool *Pool() const {
    return descriptor_pool_;
  }

  void BindUniformBuffer(uint32_t binding,
                         const class Buffer *buffer,
                         VkDeviceSize offset = 0,
                         VkDeviceSize range = VK_WHOLE_SIZE) const;

  void BindCombinedImageSampler(uint32_t binding, const struct Image *image, VkSampler sampler = VK_NULL_HANDLE) const;

  void BindStorageImage(uint32_t binding, const struct Image *image) const;

  void BindAccelerationStructure(uint32_t binding, const struct AccelerationStructure *acceleration_structure) const;

  void BindStorageBuffer(uint32_t binding,
                         const class Buffer *buffer,
                         VkDeviceSize offset = 0,
                         VkDeviceSize range = VK_WHOLE_SIZE) const;

  void BindStorageBuffers(uint32_t binding,
                          const std::vector<const class Buffer *> &buffers,
                          const std::vector<VkDeviceSize> &offsets = {},
                          const std::vector<VkDeviceSize> &ranges = {}) const;

  void BindSampledImages(uint32_t binding, const std::vector<const struct Image *> &images) const;

  void BindInputAttachment(uint32_t binding, const struct Image *image) const;

  void BindSamplers(uint32_t binding, const std::vector<VkSampler> &samplers) const;

 private:
  const class DescriptorPool *descriptor_pool_{};
  VkDescriptorSet set_{};
};
}  // namespace CD::vulkan
