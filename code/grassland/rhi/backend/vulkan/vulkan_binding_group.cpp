#include "grassland/rhi/backend/vulkan/vulkan_binding_group.h"

#include <algorithm>
#include <limits>

#include "grassland/rhi/backend/vulkan/vulkan_allocated_image.h"
#include "grassland/rhi/backend/vulkan/vulkan_buffer.h"
#include "grassland/rhi/backend/vulkan/vulkan_device.h"

namespace grassland::rhi::backend::vulkan {
namespace {
constexpr VkDescriptorType descriptor_types[] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                 VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                 VK_DESCRIPTOR_TYPE_SAMPLER};

size_t Kind(BindingType type) {
  const size_t kinds[] = {0, 1, 1, 2, 3, 4};
  return kinds[type];
}

VkDescriptorType Type(BindingType type) {
  return descriptor_types[Kind(type)];
}
}  // namespace

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
      auto &count = pattern[Kind(binding.type)];
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
          sizes.push_back({descriptor_types[i], pattern[i] * fresh->capacity});
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

VkDescriptorSetLayout CreateBindingLayout(VkDevice device, const BindingLayoutDesc &layout) {
  std::vector<VkDescriptorSetLayoutBinding> bindings;
  for (auto b : layout.bindings) {
    VkShaderStageFlags stages = 0;
    if (b.visibility & SHADER_VISIBILITY_VERTEX)
      stages |= VK_SHADER_STAGE_VERTEX_BIT;
    if (b.visibility & SHADER_VISIBILITY_FRAGMENT)
      stages |= VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings.push_back({b.binding, Type(b.type), b.count, stages, nullptr});
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
  try {
    if (!ValidateBindingLayout(layout))
      return nullptr;
    if (!ValidatePipelineBindings(device->NativePhysicalDevice(), {{layout}}))
      return nullptr;
    auto version = ObjectPtr<Version>::Adopt(Version::Create(device, layout));
    if (!version)
      return nullptr;
    return new VulkanBindingGroup(device, std::move(version));
  } catch (const std::bad_alloc &) {
    return nullptr;
  }
}

VulkanBindingGroup::Version::Version(VulkanDevice *device, const BindingLayoutDesc &layout)
    : BindingGroup::Version(device, layout) {
}

VulkanBindingGroup::Version::~Version() {
  auto device = static_cast<VulkanDevice *>(device_)->NativeDevice();
  lease_ = {};
  for (auto view : views_)
    if (view)
      vkDestroyImageView(device, view, nullptr);
  for (auto sampler : samplers_)
    if (sampler)
      vkDestroySampler(device, sampler, nullptr);
  if (native_layout_)
    vkDestroyDescriptorSetLayout(device, native_layout_, nullptr);
}

VkDescriptorSet VulkanBindingGroup::Version::Set() const {
  return lease_->Set();
}

VulkanBindingGroup::Version *VulkanBindingGroup::Version::Create(VulkanDevice *device,
                                                                 const BindingLayoutDesc &layout) {
  auto version = std::unique_ptr<Version>(new Version(device, layout));
  version->views_.resize(version->SlotCount());
  version->samplers_.resize(version->SlotCount());
  version->native_layout_ = CreateBindingLayout(device->NativeDevice(), layout);
  if (!version->native_layout_)
    return nullptr;
  version->lease_ = device->DescriptorPool().Acquire(version->native_layout_, layout);
  if (!version->lease_)
    return nullptr;
  return version.release();
}

ObjectPtr<BindingGroup::Version> VulkanBindingGroup::Version::CreateEmpty() const {
  return ObjectPtr<Version>::Adopt(Create(static_cast<VulkanDevice *>(device_), layout_));
}

bool VulkanBindingGroup::Version::WriteDescriptors(size_t offset,
                                                   const BindingDesc &binding,
                                                   const std::vector<BufferView> &buffers) {
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
  write.dstArrayElement = ArrayElement(offset);
  write.descriptorCount = uint32_t(infos.size());
  write.descriptorType = uniform ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  write.pBufferInfo = infos.data();
  vkUpdateDescriptorSets(device->NativeDevice(), 1, &write, 0, nullptr);
  return true;
}

bool VulkanBindingGroup::Version::WriteDescriptors(size_t offset,
                                                   const BindingDesc &binding,
                                                   const std::vector<IImage *> &images) {
  for (auto image : images)
    if (!dynamic_cast<VulkanAllocatedImage *>(image))
      return false;
  auto native = static_cast<VulkanDevice *>(device_)->NativeDevice();
  // Stage all native handles before changing the set or any previous slot.
  std::vector<VkDescriptorImageInfo> infos(images.size());
  for (size_t i = 0; i < images.size(); ++i) {
    auto image = images[i];
    VkImageViewCreateInfo create{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    create.image = static_cast<VulkanImage *>(image)->NativeImage();
    create.viewType = VK_IMAGE_VIEW_TYPE_2D;
    create.format = NativeImageFormat(image->GetDesc().format);
    create.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (!Check(vkCreateImageView(native, &create, nullptr, &infos[i].imageView), "Create binding view")) {
      for (const auto &info : infos)
        if (info.imageView)
          vkDestroyImageView(native, info.imageView, nullptr);
      return false;
    }
    infos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  }
  VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  write.dstSet = Set();
  write.dstBinding = binding.binding;
  write.dstArrayElement = ArrayElement(offset);
  write.descriptorCount = uint32_t(infos.size());
  write.descriptorType =
      binding.type == BINDING_TEXTURE2D ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  write.pImageInfo = infos.data();
  vkUpdateDescriptorSets(native, 1, &write, 0, nullptr);
  for (size_t i = 0; i < infos.size(); ++i) {
    if (views_[offset + i])
      vkDestroyImageView(native, views_[offset + i], nullptr);
    views_[offset + i] = infos[i].imageView;
  }
  return true;
}

bool VulkanBindingGroup::Version::WriteDescriptors(size_t offset,
                                                   const BindingDesc &binding,
                                                   const std::vector<SamplerDesc> &samplers) {
  auto native = static_cast<VulkanDevice *>(device_)->NativeDevice();
  std::vector<VkDescriptorImageInfo> infos(samplers.size());
  const VkSamplerAddressMode modes[] = {VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                        VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT};
  for (size_t i = 0; i < samplers.size(); ++i) {
    const auto &sampler = samplers[i];
    VkSamplerCreateInfo create{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    create.magFilter = create.minFilter =
        sampler.filter == SAMPLER_FILTER_LINEAR ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    create.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    create.addressModeU = modes[sampler.address_u];
    create.addressModeV = create.addressModeW = modes[sampler.address_v];
    create.maxLod = VK_LOD_CLAMP_NONE;
    if (!Check(vkCreateSampler(native, &create, nullptr, &infos[i].sampler), "Create sampler")) {
      for (const auto &info : infos)
        if (info.sampler)
          vkDestroySampler(native, info.sampler, nullptr);
      return false;
    }
  }
  VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  write.dstSet = Set();
  write.dstBinding = binding.binding;
  write.dstArrayElement = ArrayElement(offset);
  write.descriptorCount = uint32_t(infos.size());
  write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
  write.pImageInfo = infos.data();
  vkUpdateDescriptorSets(native, 1, &write, 0, nullptr);
  for (size_t i = 0; i < infos.size(); ++i) {
    if (samplers_[offset + i])
      vkDestroySampler(native, samplers_[offset + i], nullptr);
    samplers_[offset + i] = infos[i].sampler;
  }
  return true;
}

bool ValidatePipelineBindings(VkPhysicalDevice device, const PipelineLayoutDesc &layout) {
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(device, &properties);
  const auto &limits = properties.limits;
  if (layout.groups.size() > limits.maxBoundDescriptorSets)
    return false;
  uint64_t totals[5]{}, stages[2][5]{};
  for (const auto &group : layout.groups)
    for (auto b : group.bindings) {
      auto kind = Kind(b.type);
      totals[kind] += b.count;
      for (int i = 0; i < 2; ++i)
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
    for (int i = 0; i < 2; ++i)
      if (stages[i][kind] > stage_limits[kind])
        return false;
  }
  for (int i = 0; i < 2; ++i)
    if (stages[i][0] + stages[i][1] + stages[i][2] + stages[i][3] > limits.maxPerStageResources)
      return false;
  VkPhysicalDeviceFeatures features{};
  vkGetPhysicalDeviceFeatures(device, &features);
  for (const auto &group : layout.groups)
    for (auto b : group.bindings)
      if (b.type == BINDING_RW_STORAGE_BUFFER || b.type == BINDING_RW_TEXTURE2D) {
        if ((b.visibility & SHADER_VISIBILITY_VERTEX) && !features.vertexPipelineStoresAndAtomics)
          return false;
        if ((b.visibility & SHADER_VISIBILITY_FRAGMENT) && !features.fragmentStoresAndAtomics)
          return false;
      }
  return true;
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
