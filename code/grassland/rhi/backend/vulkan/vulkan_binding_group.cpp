#include "grassland/rhi/backend/vulkan/vulkan_binding_group.h"

#include <algorithm>
#include <limits>

#include "grassland/rhi/backend/vulkan/vulkan_allocated_image.h"
#include "grassland/rhi/backend/vulkan/vulkan_buffer.h"
#include "grassland/rhi/backend/vulkan/vulkan_device.h"
#include "grassland/rhi/backend/vulkan/vulkan_sampler.h"

namespace grassland::rhi::backend::vulkan {

VkDescriptorSetLayout CreateBindingLayout(VkDevice device, const BindingLayoutDesc &layout) {
  std::vector<VkDescriptorSetLayoutBinding> bindings;
  for (auto b : layout.bindings) {
    VkShaderStageFlags stages = 0;
    if (b.visibility & SHADER_VISIBILITY_VERTEX)
      stages |= VK_SHADER_STAGE_VERTEX_BIT;
    if (b.visibility & SHADER_VISIBILITY_FRAGMENT)
      stages |= VK_SHADER_STAGE_FRAGMENT_BIT;
    if (b.visibility & SHADER_VISIBILITY_COMPUTE)
      stages |= VK_SHADER_STAGE_COMPUTE_BIT;
    bindings.push_back({b.binding, NativeDescriptorType(b.type), b.count, stages, nullptr});
  }
  VkDescriptorSetLayoutCreateInfo create{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  create.bindingCount = uint32_t(bindings.size());
  create.pBindings = bindings.data();
  VkDescriptorSetLayoutSupport support{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_SUPPORT};
  vkGetDescriptorSetLayoutSupport(device, &create, &support);
  if (!support.supported)
    return {};
  VkDescriptorSetLayout result{};
  if (!Check(vkCreateDescriptorSetLayout(device, &create, nullptr, &result), "Create binding layout"))
    return {};
  return result;
}

VulkanBindingGroup::VulkanBindingGroup(VulkanDevice *device, ObjectPtr<Version> version)
    : BindingGroup(device, std::move(version)) {
}

VulkanBindingGroup *VulkanBindingGroup::Create(VulkanDevice *device, const BindingLayoutDesc &layout) {
  if (!ValidateBindingLayout(layout))
    return nullptr;
  if (!ValidatePipelineBindings(device->NativePhysicalDevice(), {{layout}}))
    return nullptr;
  auto version = ObjectPtr<Version>::Adopt(Version::Create(device, layout));
  if (!version)
    return nullptr;
  return new VulkanBindingGroup(device, std::move(version));
}

VulkanBindingGroup::Version::Version(VulkanDevice *device, const BindingLayoutDesc &layout)
    : BindingGroup::Version(device, layout) {
  for (size_t i = 0; i < layout_.bindings.size(); ++i) {
    const auto &binding = layout_.bindings[i];
    initialized_.emplace_back(binding.count, false);
  }
}

VulkanBindingGroup::Version::~Version() {
  auto device = static_cast<VulkanDevice *>(device_)->NativeDevice();
  lease_ = {};
  if (native_layout_)
    vkDestroyDescriptorSetLayout(device, native_layout_, nullptr);
}

VkDescriptorSet VulkanBindingGroup::Version::Set() const {
  return lease_->Set();
}

VulkanBindingGroup::Version *VulkanBindingGroup::Version::Create(VulkanDevice *device,
                                                                 const BindingLayoutDesc &layout) {
  auto version = std::unique_ptr<Version>(new Version(device, layout));
  version->native_layout_ = CreateBindingLayout(device->NativeDevice(), layout);
  if (!version->native_layout_)
    return nullptr;
  version->lease_ = device->DescriptorPool().Acquire(version->native_layout_, layout);
  if (!version->lease_)
    return nullptr;
  return version.release();
}

ObjectPtr<BindingGroup::Version> VulkanBindingGroup::Version::Clone() const {
  auto device = static_cast<VulkanDevice *>(device_);
  auto copy = ObjectPtr<Version>::Adopt(Create(device, layout_));
  if (!copy)
    return {};
  copy->references_ = references_;
  copy->initialized_ = initialized_;
  std::vector<VkCopyDescriptorSet> copies;
  for (size_t i = 0; i < layout_.bindings.size(); ++i) {
    const auto &binding = layout_.bindings[i];
    uint32_t element = 0;
    while (element < binding.count) {
      if (!initialized_[i][element]) {
        ++element;
        continue;
      }
      const auto first = element;
      while (element < binding.count && initialized_[i][element])
        ++element;
      VkCopyDescriptorSet range{VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET};
      range.srcSet = Set();
      range.dstSet = copy->Set();
      range.srcBinding = range.dstBinding = binding.binding;
      range.srcArrayElement = range.dstArrayElement = first;
      range.descriptorCount = element - first;
      copies.push_back(range);
    }
  }
  vkUpdateDescriptorSets(device->NativeDevice(), 0, nullptr, uint32_t(copies.size()), copies.data());
  return copy;
}

bool VulkanBindingGroup::Version::IsInitialized(size_t binding_index, uint32_t element) const {
  return initialized_[binding_index][element];
}

bool VulkanBindingGroup::Version::WriteDescriptors(size_t binding_index,
                                                   uint32_t first_element,
                                                   const std::vector<BufferView> &buffers) {
  const auto &binding = layout_.bindings[binding_index];
  auto device = static_cast<VulkanDevice *>(device_);
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(device->NativePhysicalDevice(), &properties);
  const bool uniform = binding.type == BINDING_UNIFORM_BUFFER;
  const auto alignment =
      uniform ? properties.limits.minUniformBufferOffsetAlignment : properties.limits.minStorageBufferOffsetAlignment;
  const auto limit = uniform ? properties.limits.maxUniformBufferRange : properties.limits.maxStorageBufferRange;
  std::vector<VkDescriptorBufferInfo> infos;
  infos.reserve(buffers.size());
  for (const auto &view : buffers) {
    if (view.offset % alignment || view.size > limit)
      return false;
    infos.push_back({static_cast<VulkanBuffer *>(view.buffer)->NativeBuffer(), view.offset, view.size});
  }
  VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  write.dstSet = Set();
  write.dstBinding = binding.binding;
  write.dstArrayElement = first_element;
  write.descriptorCount = uint32_t(infos.size());
  write.descriptorType = uniform ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  write.pBufferInfo = infos.data();
  vkUpdateDescriptorSets(device->NativeDevice(), 1, &write, 0, nullptr);
  for (size_t i = 0; i < buffers.size(); ++i)
    initialized_[binding_index][first_element + i] = true;
  return true;
}

bool VulkanBindingGroup::Version::WriteDescriptors(size_t binding_index,
                                                   uint32_t first_element,
                                                   const std::vector<IImage *> &images) {
  const auto &binding = layout_.bindings[binding_index];
  for (auto image : images)
    if (!dynamic_cast<VulkanAllocatedImage *>(image))
      return false;
  auto native = static_cast<VulkanDevice *>(device_)->NativeDevice();
  std::vector<VkDescriptorImageInfo> infos(images.size());
  for (size_t i = 0; i < images.size(); ++i) {
    infos[i].imageView = static_cast<VulkanImage *>(images[i])->BindingView();
    infos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  }
  VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  write.dstSet = Set();
  write.dstBinding = binding.binding;
  write.dstArrayElement = first_element;
  write.descriptorCount = uint32_t(infos.size());
  write.descriptorType =
      binding.type == BINDING_TEXTURE2D ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  write.pImageInfo = infos.data();
  vkUpdateDescriptorSets(native, 1, &write, 0, nullptr);
  for (size_t i = 0; i < infos.size(); ++i) {
    initialized_[binding_index][first_element + i] = true;
  }
  return true;
}

bool VulkanBindingGroup::Version::WriteDescriptors(size_t binding_index,
                                                   uint32_t first_element,
                                                   const std::vector<ISampler *> &samplers) {
  const auto &binding = layout_.bindings[binding_index];
  auto native = static_cast<VulkanDevice *>(device_)->NativeDevice();
  std::vector<VkDescriptorImageInfo> infos(samplers.size());
  for (size_t i = 0; i < samplers.size(); ++i)
    infos[i].sampler = static_cast<VulkanSampler *>(samplers[i])->NativeSampler();
  VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  write.dstSet = Set();
  write.dstBinding = binding.binding;
  write.dstArrayElement = first_element;
  write.descriptorCount = uint32_t(infos.size());
  write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
  write.pImageInfo = infos.data();
  vkUpdateDescriptorSets(native, 1, &write, 0, nullptr);
  for (size_t i = 0; i < infos.size(); ++i) {
    initialized_[binding_index][first_element + i] = true;
  }
  return true;
}

bool ValidatePipelineBindings(VkPhysicalDevice device, const PipelineLayoutDesc &layout) {
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(device, &properties);
  const auto &limits = properties.limits;
  if (layout.groups.size() > limits.maxBoundDescriptorSets)
    return false;
  uint64_t totals[5]{}, stages[3][5]{};
  for (const auto &group : layout.groups)
    for (auto b : group.bindings) {
      auto kind = DescriptorTypeIndex(b.type);
      totals[kind] += b.count;
      for (int i = 0; i < 3; ++i)
        if (b.visibility & (1 << i))
          stages[i][kind] += b.count;
    }
  const uint32_t set_limits[] = {limits.maxDescriptorSetUniformBuffers, limits.maxDescriptorSetStorageBuffers,
                                 limits.maxDescriptorSetSampledImages, limits.maxDescriptorSetStorageImages,
                                 limits.maxDescriptorSetSamplers};
  const uint32_t stage_limits[] = {limits.maxPerStageDescriptorUniformBuffers,
                                   limits.maxPerStageDescriptorStorageBuffers,
                                   limits.maxPerStageDescriptorSampledImages, limits.maxPerStageDescriptorStorageImages,
                                   limits.maxPerStageDescriptorSamplers};
  for (int kind = 0; kind < 5; ++kind) {
    if (totals[kind] > set_limits[kind])
      return false;
    for (int i = 0; i < 3; ++i)
      if (stages[i][kind] > stage_limits[kind])
        return false;
  }
  for (int i = 0; i < 3; ++i)
    if (stages[i][0] + stages[i][1] + stages[i][2] + stages[i][3] > limits.maxPerStageResources)
      return false;
  VkPhysicalDeviceFeatures features{};
  vkGetPhysicalDeviceFeatures(device, &features);
  for (const auto &group : layout.groups)
    for (auto b : group.bindings)
      if (b.type == BINDING_RW_STRUCTURED_BUFFER || b.type == BINDING_RW_BYTE_ADDRESS_BUFFER ||
          b.type == BINDING_RW_TEXTURE2D) {
        if ((b.visibility & SHADER_VISIBILITY_VERTEX) && !features.vertexPipelineStoresAndAtomics)
          return false;
        if ((b.visibility & SHADER_VISIBILITY_FRAGMENT) && !features.fragmentStoresAndAtomics)
          return false;
      }
  return true;
}

}  // namespace grassland::rhi::backend::vulkan
