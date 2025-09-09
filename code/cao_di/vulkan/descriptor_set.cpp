#include "cao_di/vulkan/descriptor_set.h"

#include "cao_di/vulkan/buffer.h"
#include "cao_di/vulkan/image.h"
#include "cao_di/vulkan/raytracing/acceleration_structure.h"

namespace CD::vulkan {
DescriptorSet::DescriptorSet(const struct DescriptorPool *descriptor_pool, VkDescriptorSet set)
    : descriptor_pool_(descriptor_pool), set_(set) {
}

DescriptorSet::~DescriptorSet() {
  vkFreeDescriptorSets(descriptor_pool_->Device()->Handle(), descriptor_pool_->Handle(), 1, &set_);
}

void DescriptorSet::BindUniformBuffer(uint32_t binding,
                                      const struct Buffer *buffer,
                                      VkDeviceSize offset,
                                      VkDeviceSize range) const {
  VkDescriptorBufferInfo buffer_info{};
  buffer_info.buffer = buffer->Handle();
  buffer_info.offset = offset;
  buffer_info.range = range ? range : buffer->Size();

  VkWriteDescriptorSet write_descriptor_set{};
  write_descriptor_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write_descriptor_set.dstSet = set_;
  write_descriptor_set.dstBinding = binding;
  write_descriptor_set.dstArrayElement = 0;
  write_descriptor_set.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  write_descriptor_set.descriptorCount = 1;
  write_descriptor_set.pBufferInfo = &buffer_info;

  vkUpdateDescriptorSets(descriptor_pool_->Device()->Handle(), 1, &write_descriptor_set, 0, nullptr);
}

void DescriptorSet::BindCombinedImageSampler(uint32_t binding, const struct Image *image, VkSampler sampler) const {
  VkDescriptorImageInfo image_info{};
  image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  image_info.imageView = image->ImageView();
  image_info.sampler = sampler;

  VkWriteDescriptorSet write_descriptor_set{};
  write_descriptor_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write_descriptor_set.dstSet = set_;
  write_descriptor_set.dstBinding = binding;
  write_descriptor_set.dstArrayElement = 0;
  write_descriptor_set.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  write_descriptor_set.descriptorCount = 1;
  write_descriptor_set.pImageInfo = &image_info;

  vkUpdateDescriptorSets(descriptor_pool_->Device()->Handle(), 1, &write_descriptor_set, 0, nullptr);
}

void DescriptorSet::BindStorageImage(uint32_t binding, const struct Image *image) const {
  VkDescriptorImageInfo image_info{};
  image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  image_info.imageView = image->ImageView();
  image_info.sampler = VK_NULL_HANDLE;

  VkWriteDescriptorSet write_descriptor_set{};
  write_descriptor_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write_descriptor_set.dstSet = set_;
  write_descriptor_set.dstBinding = binding;
  write_descriptor_set.dstArrayElement = 0;
  write_descriptor_set.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  write_descriptor_set.descriptorCount = 1;
  write_descriptor_set.pImageInfo = &image_info;

  vkUpdateDescriptorSets(descriptor_pool_->Device()->Handle(), 1, &write_descriptor_set, 0, nullptr);
}

void DescriptorSet::BindAccelerationStructure(uint32_t binding,
                                              const struct AccelerationStructure *acceleration_structure) const {
  VkAccelerationStructureKHR as = acceleration_structure->Handle();

  VkWriteDescriptorSetAccelerationStructureKHR write_descriptor_set{};
  write_descriptor_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
  write_descriptor_set.accelerationStructureCount = 1;
  write_descriptor_set.pAccelerationStructures = &as;

  VkWriteDescriptorSet write_descriptor_set_info{};
  write_descriptor_set_info.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write_descriptor_set_info.pNext = &write_descriptor_set;
  write_descriptor_set_info.dstSet = set_;
  write_descriptor_set_info.dstBinding = binding;
  write_descriptor_set_info.dstArrayElement = 0;
  write_descriptor_set_info.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
  write_descriptor_set_info.descriptorCount = 1;

  vkUpdateDescriptorSets(descriptor_pool_->Device()->Handle(), 1, &write_descriptor_set_info, 0, nullptr);
}

void DescriptorSet::BindStorageBuffers(uint32_t binding,
                                       const std::vector<const struct Buffer *> &buffers,
                                       const std::vector<VkDeviceSize> &offsets,
                                       const std::vector<VkDeviceSize> &ranges) const {
  std::vector<VkDescriptorBufferInfo> buffer_infos;
  buffer_infos.reserve(buffers.size());
  for (size_t i = 0; i < buffers.size(); ++i) {
    VkDescriptorBufferInfo buffer_info{};
    buffer_info.buffer = buffers[i]->Handle();
    buffer_info.offset = i < offsets.size() ? offsets[i] : 0;
    buffer_info.range = i < ranges.size() ? ranges[i] : buffers[i]->Size();
    buffer_infos.push_back(buffer_info);
  }

  VkWriteDescriptorSet write_descriptor_set{};
  write_descriptor_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write_descriptor_set.dstSet = set_;
  write_descriptor_set.dstBinding = binding;
  write_descriptor_set.dstArrayElement = 0;
  write_descriptor_set.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  write_descriptor_set.descriptorCount = static_cast<uint32_t>(buffer_infos.size());
  write_descriptor_set.pBufferInfo = buffer_infos.data();

  vkUpdateDescriptorSets(descriptor_pool_->Device()->Handle(), 1, &write_descriptor_set, 0, nullptr);
}

void DescriptorSet::BindSampledImages(uint32_t binding, const std::vector<const struct Image *> &images) const {
  std::vector<VkDescriptorImageInfo> image_infos;
  image_infos.reserve(images.size());
  for (size_t i = 0; i < images.size(); i++) {
    VkDescriptorImageInfo image_info{};
    image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    image_info.imageView = images[i]->ImageView();
    image_info.sampler = VK_NULL_HANDLE;
    image_infos.push_back(image_info);
  }

  VkWriteDescriptorSet write_descriptor_set{};
  write_descriptor_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write_descriptor_set.dstSet = set_;
  write_descriptor_set.dstBinding = binding;
  write_descriptor_set.dstArrayElement = 0;
  write_descriptor_set.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
  write_descriptor_set.descriptorCount = static_cast<uint32_t>(images.size());
  write_descriptor_set.pImageInfo = image_infos.data();

  vkUpdateDescriptorSets(descriptor_pool_->Device()->Handle(), 1, &write_descriptor_set, 0, nullptr);
}

void DescriptorSet::BindSamplers(uint32_t binding, const std::vector<VkSampler> &samplers) const {
  std::vector<VkDescriptorImageInfo> image_infos;
  image_infos.reserve(samplers.size());
  for (size_t i = 0; i < samplers.size(); i++) {
    VkDescriptorImageInfo image_info{};
    image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    image_info.imageView = VK_NULL_HANDLE;
    image_info.sampler = samplers[i];
    image_infos.push_back(image_info);
  }

  VkWriteDescriptorSet write_descriptor_set{};
  write_descriptor_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write_descriptor_set.dstSet = set_;
  write_descriptor_set.dstBinding = binding;
  write_descriptor_set.dstArrayElement = 0;
  write_descriptor_set.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
  write_descriptor_set.descriptorCount = static_cast<uint32_t>(samplers.size());
  write_descriptor_set.pImageInfo = image_infos.data();

  vkUpdateDescriptorSets(descriptor_pool_->Device()->Handle(), 1, &write_descriptor_set, 0, nullptr);
}

void DescriptorSet::BindStorageBuffer(uint32_t binding,
                                      const struct Buffer *buffer,
                                      VkDeviceSize offset,
                                      VkDeviceSize range) const {
  BindStorageBuffers(binding, {buffer}, {offset}, {range});
}

void DescriptorSet::BindInputAttachment(uint32_t binding, const struct Image *image) const {
  VkDescriptorImageInfo image_info{};
  image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  image_info.imageView = image->ImageView();
  image_info.sampler = VK_NULL_HANDLE;

  VkWriteDescriptorSet write_descriptor_set{};
  write_descriptor_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write_descriptor_set.dstSet = set_;
  write_descriptor_set.dstBinding = binding;
  write_descriptor_set.dstArrayElement = 0;
  write_descriptor_set.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
  write_descriptor_set.descriptorCount = 1;
  write_descriptor_set.pImageInfo = &image_info;

  vkUpdateDescriptorSets(descriptor_pool_->Device()->Handle(), 1, &write_descriptor_set, 0, nullptr);
}

}  // namespace CD::vulkan
