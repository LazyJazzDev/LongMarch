#include "grassland/rhi/backend/vulkan/vulkan_binding_group.h"

#include <algorithm>
#include <limits>
#include <set>

#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/backend_buffer.h"
#include "grassland/rhi/backend/backend_device.h"
#include "grassland/rhi/backend/backend_image.h"
#include "grassland/rhi/backend/backend_sampler.h"
#include "grassland/rhi/backend/vulkan/vulkan_acceleration_structure.h"
#include "grassland/rhi/backend/vulkan/vulkan_allocated_image.h"
#include "grassland/rhi/backend/vulkan/vulkan_binding_group_impl.h"
#include "grassland/rhi/backend/vulkan/vulkan_buffer.h"
#include "grassland/rhi/backend/vulkan/vulkan_device.h"
#include "grassland/rhi/backend/vulkan/vulkan_sampler.h"
#include "grassland/rhi/rhi_acceleration_structure.h"

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
    if (b.visibility == SHADER_VISIBILITY_ALL)
      stages = VK_SHADER_STAGE_ALL;
    else {
      if (b.visibility & SHADER_VISIBILITY_RAY_GENERATION)
        stages |= VK_SHADER_STAGE_RAYGEN_BIT_KHR;
      if (b.visibility & SHADER_VISIBILITY_MISS)
        stages |= VK_SHADER_STAGE_MISS_BIT_KHR;
      if (b.visibility & SHADER_VISIBILITY_CLOSEST_HIT)
        stages |= VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
      if (b.visibility & SHADER_VISIBILITY_ANY_HIT)
        stages |= VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
      if (b.visibility & SHADER_VISIBILITY_INTERSECTION)
        stages |= VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
      if (b.visibility & SHADER_VISIBILITY_CALLABLE)
        stages |= VK_SHADER_STAGE_CALLABLE_BIT_KHR;
    }
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

VulkanBindingGroupImpl::VulkanBindingGroupImpl(IVulkanDevice *device, ObjectPtr<Version> version) {
  InitializeBindingGroup(device, std::move(version));
}

IBindingGroup *VulkanBindingGroupImpl::Create(IVulkanDevice *device, const BindingLayoutDesc &layout) {
  if (!ValidateBindingLayout(layout))
    return nullptr;
  if (!ValidatePipelineBindings(device->NativePhysicalDevice(), {{layout}}))
    return nullptr;
  auto version = ObjectPtr<Version>::Adopt(Version::Create(device, layout));
  if (!version)
    return nullptr;
  return new VulkanBindingGroupImpl(device, std::move(version));
}

VulkanBindingVersionImpl::VulkanBindingVersionImpl(IVulkanDevice *device, const BindingLayoutDesc &layout)
    : device_(device) {
  InitializeBindingVersion(layout);

  for (size_t i = 0; i < layout_.bindings.size(); ++i) {
    const auto &binding = layout_.bindings[i];
    initialized_.emplace_back(binding.count, false);
  }
}

VulkanBindingVersionImpl::~VulkanBindingVersionImpl() {
  auto device = device_->NativeDevice();
  lease_ = {};
  if (native_layout_)
    vkDestroyDescriptorSetLayout(device, native_layout_, nullptr);
}

VkDescriptorSet VulkanBindingVersionImpl::Set() const {
  return lease_->Set();
}

VulkanBindingVersionImpl *VulkanBindingVersionImpl::Create(IVulkanDevice *device, const BindingLayoutDesc &layout) {
  auto version = std::unique_ptr<VulkanBindingVersionImpl>(new VulkanBindingVersionImpl(device, layout));
  version->native_layout_ = CreateBindingLayout(device->NativeDevice(), layout);
  if (!version->native_layout_)
    return nullptr;
  version->lease_ = device->DescriptorPool().Acquire(version->native_layout_, layout);
  if (!version->lease_)
    return nullptr;
  return version.release();
}

ObjectPtr<IBackendBindingVersion> VulkanBindingVersionImpl::Clone() const {
  auto copy = ObjectPtr<VulkanBindingVersionImpl>::Adopt(Create(device_, layout_));
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
  vkUpdateDescriptorSets(device_->NativeDevice(), 0, nullptr, uint32_t(copies.size()), copies.data());
  return copy;
}

bool VulkanBindingVersionImpl::IsInitialized(size_t binding_index, uint32_t element) const {
  return initialized_[binding_index][element];
}

bool VulkanBindingVersionImpl::WriteDescriptors(size_t binding_index,
                                                uint32_t first_element,
                                                const std::vector<BufferView> &buffers) {
  const auto &binding = layout_.bindings[binding_index];
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(device_->NativePhysicalDevice(), &properties);
  const bool uniform = binding.type == BINDING_UNIFORM_BUFFER;
  const auto alignment =
      uniform ? properties.limits.minUniformBufferOffsetAlignment : properties.limits.minStorageBufferOffsetAlignment;
  const auto limit = uniform ? properties.limits.maxUniformBufferRange : properties.limits.maxStorageBufferRange;
  std::vector<VkDescriptorBufferInfo> infos;
  infos.reserve(buffers.size());
  for (const auto &view : buffers) {
    if (view.offset % alignment || view.size > limit)
      return false;
    infos.push_back({rhi::QueryInterfacePtr<IVulkanBuffer>(view.buffer).Get()->NativeBuffer(), view.offset, view.size});
  }
  VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  write.dstSet = Set();
  write.dstBinding = binding.binding;
  write.dstArrayElement = first_element;
  write.descriptorCount = uint32_t(infos.size());
  write.descriptorType = uniform ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  write.pBufferInfo = infos.data();
  vkUpdateDescriptorSets(device_->NativeDevice(), 1, &write, 0, nullptr);
  for (size_t i = 0; i < buffers.size(); ++i)
    initialized_[binding_index][first_element + i] = true;
  return true;
}

bool VulkanBindingVersionImpl::WriteDescriptors(size_t binding_index,
                                                uint32_t first_element,
                                                const std::vector<IImage *> &images) {
  const auto &binding = layout_.bindings[binding_index];
  std::vector<IVulkanImage *> native_images;
  native_images.reserve(images.size());
  for (auto image : images) {
    auto native_image = rhi::QueryInterfacePtr<IVulkanImage>(image);
    if (!native_image)
      return false;
    native_images.push_back(native_image.Get());
  }
  auto native = device_->NativeDevice();
  std::vector<VkDescriptorImageInfo> infos(images.size());
  for (size_t i = 0; i < images.size(); ++i) {
    infos[i].imageView = native_images[i]->BindingView();
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

bool VulkanBindingVersionImpl::WriteDescriptors(size_t binding_index,
                                                uint32_t first_element,
                                                const std::vector<ISampler *> &samplers) {
  const auto &binding = layout_.bindings[binding_index];
  auto native = device_->NativeDevice();
  std::vector<VkDescriptorImageInfo> infos(samplers.size());
  for (size_t i = 0; i < samplers.size(); ++i)
    infos[i].sampler = rhi::QueryInterfacePtr<IVulkanSampler>(samplers[i]).Get()->NativeSampler();
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
  VkPhysicalDeviceAccelerationStructurePropertiesKHR as{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
  VkPhysicalDeviceProperties2 extended{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &as};
  vkGetPhysicalDeviceProperties2(device, &extended);
  uint64_t totals[6]{}, stages[9][6]{};
  for (const auto &group : layout.groups)
    for (auto b : group.bindings) {
      auto kind = DescriptorTypeIndex(b.type);
      totals[kind] += b.count;
      for (int i = 0; i < 9; ++i)
        if (b.visibility & (1 << i))
          stages[i][kind] += b.count;
    }
  const uint32_t set_limits[] = {limits.maxDescriptorSetUniformBuffers, limits.maxDescriptorSetStorageBuffers,
                                 limits.maxDescriptorSetSampledImages,  limits.maxDescriptorSetStorageImages,
                                 limits.maxDescriptorSetSamplers,       as.maxDescriptorSetAccelerationStructures};
  const uint32_t stage_limits[] = {
      limits.maxPerStageDescriptorUniformBuffers, limits.maxPerStageDescriptorStorageBuffers,
      limits.maxPerStageDescriptorSampledImages,  limits.maxPerStageDescriptorStorageImages,
      limits.maxPerStageDescriptorSamplers,       as.maxPerStageDescriptorAccelerationStructures};
  for (int kind = 0; kind < 6; ++kind) {
    if (totals[kind] > set_limits[kind])
      return false;
    for (int i = 0; i < 9; ++i)
      if (stages[i][kind] > stage_limits[kind])
        return false;
  }
  for (int i = 0; i < 9; ++i)
    if (stages[i][0] + stages[i][1] + stages[i][2] + stages[i][3] + stages[i][5] > limits.maxPerStageResources)
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

bool VulkanBindingVersionImpl::WriteDescriptors(size_t binding_index,
                                                uint32_t first_element,
                                                const std::vector<ITopLevelAccelerationStructure *> &values) {
  std::vector<VkAccelerationStructureKHR> handles;
  handles.reserve(values.size());
  for (auto value : values) {
    auto native = rhi::QueryInterfacePtr<IVulkanTopLevelAccelerationStructure>(value).Get();
    if (!native || native->Owner() != device_)
      return false;
    handles.push_back(native->NativeStructure());
  }
  VkWriteDescriptorSetAccelerationStructureKHR as{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
  as.accelerationStructureCount = uint32_t(handles.size());
  as.pAccelerationStructures = handles.data();
  VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, &as};
  write.dstSet = Set();
  write.dstBinding = layout_.bindings[binding_index].binding;
  write.dstArrayElement = first_element;
  write.descriptorCount = uint32_t(handles.size());
  write.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
  vkUpdateDescriptorSets(device_->NativeDevice(), 1, &write, 0, nullptr);
  for (size_t i = 0; i < values.size(); ++i)
    initialized_[binding_index][first_element + i] = true;
  return true;
}

IVulkanBindingVersion *IVulkanBindingVersion::Create(IVulkanDevice *device, const BindingLayoutDesc &layout) {
  return VulkanBindingVersionImpl::Create(device, layout);
}

IBindingGroup *CreateVulkanBindingGroup(IVulkanDevice *device, const BindingLayoutDesc &layout) {
  return VulkanBindingGroupImpl::Create(device, layout);
}

void VulkanBindingVersionImpl::InitializeBindingVersion(const BindingLayoutDesc &layout) {
  layout_ = layout;

  std::sort(layout_.bindings.begin(), layout_.bindings.end(), [](auto a, auto b) { return a.binding < b.binding; });
  references_.resize(layout_.bindings.size());
  for (size_t i = 0; i < layout_.bindings.size(); ++i)
    references_[i].resize(layout_.bindings[i].count);
}

const BindingLayoutDesc &VulkanBindingVersionImpl::GetLayout() const {
  return layout_;
}

const std::vector<std::vector<ObjectPtr<IObject>>> &VulkanBindingVersionImpl::GetReferences() const {
  return references_;
}

bool VulkanBindingVersionImpl::IsComplete() const {
  for (size_t i = 0; i < layout_.bindings.size(); ++i)
    for (uint32_t j = 0; j < layout_.bindings[i].count; ++j)
      if (!IsInitialized(i, j))
        return false;
  return true;
}

bool VulkanBindingVersionImpl::Write(size_t binding_index,
                                     uint32_t first_element,
                                     const std::vector<BufferView> &values) {
  // Retain the entire update before releasing old slots, including resource swaps.
  std::vector<ObjectPtr<IObject>> references;
  references.reserve(values.size());
  for (const auto &value : values)
    references.push_back(ObjectPtr<IObject>::Retain(value.buffer));
  if (!WriteDescriptors(binding_index, first_element, values))
    return false;
  for (size_t i = 0; i < values.size(); ++i)
    references_[binding_index][first_element + i] = std::move(references[i]);
  return true;
}

bool VulkanBindingVersionImpl::Write(size_t binding_index,
                                     uint32_t first_element,
                                     const std::vector<IImage *> &values) {
  // Retain the entire update before releasing old slots, including resource swaps.
  std::vector<ObjectPtr<IObject>> references;
  references.reserve(values.size());
  for (const auto &value : values)
    references.push_back(ObjectPtr<IObject>::Retain(value));
  if (!WriteDescriptors(binding_index, first_element, values))
    return false;
  for (size_t i = 0; i < values.size(); ++i)
    references_[binding_index][first_element + i] = std::move(references[i]);
  return true;
}

bool VulkanBindingVersionImpl::Write(size_t binding_index,
                                     uint32_t first_element,
                                     const std::vector<ISampler *> &values) {
  std::vector<ObjectPtr<IObject>> references;
  references.reserve(values.size());
  for (auto value : values)
    references.push_back(ObjectPtr<IObject>::Retain(value));
  if (!WriteDescriptors(binding_index, first_element, values))
    return false;
  for (size_t i = 0; i < values.size(); ++i)
    references_[binding_index][first_element + i] = std::move(references[i]);
  return true;
}

bool VulkanBindingVersionImpl::Write(size_t binding_index,
                                     uint32_t first_element,
                                     const std::vector<ITopLevelAccelerationStructure *> &values) {
  std::vector<ObjectPtr<IObject>> references;
  references.reserve(values.size());
  for (auto value : values)
    references.push_back(ObjectPtr<IObject>::Retain(value));
  if (!WriteDescriptors(binding_index, first_element, values))
    return false;
  for (size_t i = 0; i < values.size(); ++i)
    references_[binding_index][first_element + i] = std::move(references[i]);
  return true;
}

void VulkanBindingGroupImpl::InitializeBindingGroup(IBackendDevice *device, ObjectPtr<IBackendBindingVersion> version) {
  device_ = device;
  layout_ = version->GetLayout();
  version_ = std::move(version);
}

const BindingLayoutDesc &VulkanBindingGroupImpl::GetLayout() const {
  return layout_;
}

IBackendDevice *VulkanBindingGroupImpl::Owner() const {
  return device_;
}

IBackendBindingVersion *VulkanBindingGroupImpl::CurrentVersion() const {
  return version_.Get();
}

const BindingDesc *VulkanBindingGroupImpl::FindBinding(uint32_t binding,
                                                       uint32_t first_element,
                                                       size_t count,
                                                       size_t &binding_index) const {
  for (binding_index = 0; binding_index < layout_.bindings.size(); ++binding_index) {
    const auto &desc = layout_.bindings[binding_index];
    if (desc.binding != binding)
      continue;
    if (!count || first_element > desc.count || count > desc.count - first_element)
      return nullptr;
    return &desc;
  }
  return nullptr;
}

bool VulkanBindingGroupImpl::BindResource(uint32_t binding,
                                          const std::vector<BufferView> &buffers,
                                          uint32_t first_element) {
  size_t binding_index;
  const auto *desc = FindBinding(binding, first_element, buffers.size(), binding_index);
  if (!desc)
    return false;
  if (desc->type > BINDING_RW_BYTE_ADDRESS_BUFFER)
    return false;
  auto values = buffers;
  for (auto &view : values) {
    auto buffer = rhi::QueryInterfacePtr<IBackendBuffer>(view.buffer).Get();
    if (!buffer || buffer->Owner() != device_ || view.offset >= view.buffer->GetDesc().size)
      return false;
    if (!view.size)
      view.size = view.buffer->GetDesc().size - view.offset;
    if (view.size > view.buffer->GetDesc().size - view.offset)
      return false;
    if (desc->type == BINDING_STRUCTURED_BUFFER || desc->type == BINDING_RW_STRUCTURED_BUFFER) {
      if (view.offset % desc->stride || view.size % desc->stride)
        return false;
    } else if (desc->type == BINDING_BYTE_ADDRESS_BUFFER || desc->type == BINDING_RW_BYTE_ADDRESS_BUFFER) {
      if (view.offset % 4 || view.size % 4)
        return false;
    }
  }
  auto next = version_->IsUniquelyOwned() ? version_ : version_->Clone();
  if (!next)
    return false;
  if (!next->Write(binding_index, first_element, values))
    return false;
  version_ = std::move(next);
  return true;
}

bool VulkanBindingGroupImpl::BindResource(uint32_t binding,
                                          const std::vector<IImage *> &images,
                                          uint32_t first_element) {
  size_t binding_index;
  const auto *desc = FindBinding(binding, first_element, images.size(), binding_index);
  if (!desc)
    return false;
  if (desc->type != BINDING_TEXTURE2D && desc->type != BINDING_RW_TEXTURE2D)
    return false;
  for (auto value : images) {
    auto image = rhi::QueryInterfacePtr<IBackendImage>(value).Get();
    if (!image || !value->IsValid() || image->DeviceIdentity() != device_ || IsDepthFormat(value->GetDesc().format))
      return false;
    if (desc->type == BINDING_RW_TEXTURE2D && !image->SupportsStorage())
      return false;
  }
  auto next = version_->IsUniquelyOwned() ? version_ : version_->Clone();
  if (!next)
    return false;
  if (!next->Write(binding_index, first_element, images))
    return false;
  version_ = std::move(next);
  return true;
}

bool VulkanBindingGroupImpl::BindResource(uint32_t binding,
                                          const std::vector<ISampler *> &samplers,
                                          uint32_t first_element) {
  size_t binding_index;
  const auto *desc = FindBinding(binding, first_element, samplers.size(), binding_index);
  if (!desc)
    return false;
  if (desc->type != BINDING_SAMPLER)
    return false;
  for (auto sampler : samplers) {
    auto native = rhi::QueryInterfacePtr<IVulkanSampler>(sampler).Get();
    if (!native || native->DeviceIdentity() != device_)
      return false;
  }
  auto next = version_->IsUniquelyOwned() ? version_ : version_->Clone();
  if (!next)
    return false;
  if (!next->Write(binding_index, first_element, samplers))
    return false;
  version_ = std::move(next);
  return true;
}

bool VulkanBindingGroupImpl::BindResource(uint32_t binding,
                                          const std::vector<ITopLevelAccelerationStructure *> &structures,
                                          uint32_t first_element) {
  size_t binding_index;
  auto desc = FindBinding(binding, first_element, structures.size(), binding_index);
  if (!desc || desc->type != BINDING_ACCELERATION_STRUCTURE)
    return false;
  for (auto structure : structures)
    if (!structure)
      return false;
  auto next = version_->IsUniquelyOwned() ? version_ : version_->Clone();
  if (!next)
    return false;
  if (!next->Write(binding_index, first_element, structures))
    return false;
  version_ = std::move(next);
  return true;
}
}  // namespace grassland::rhi::backend::vulkan
