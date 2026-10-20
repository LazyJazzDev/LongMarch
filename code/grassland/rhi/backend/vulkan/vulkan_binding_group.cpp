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
  version->views_.resize(version->GetResources().size());
  version->samplers_.resize(version->GetResources().size());
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

bool VulkanBindingGroup::Version::WriteDescriptors(size_t offset, const std::vector<Resource> &resources) {
  auto device = static_cast<VulkanDevice *>(device_);
  auto native = device->NativeDevice();
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(device->NativePhysicalDevice(), &properties);
  for (auto e : resources) {
    auto binding =
        *std::find_if(layout_.bindings.begin(), layout_.bindings.end(), [e](auto b) { return b.binding == e.binding; });
    if (binding.type <= BINDING_RW_STORAGE_BUFFER) {
      bool uniform = binding.type == BINDING_UNIFORM_BUFFER;
      auto alignment = uniform ? properties.limits.minUniformBufferOffsetAlignment
                               : properties.limits.minStorageBufferOffsetAlignment;
      auto limit = uniform ? properties.limits.maxUniformBufferRange : properties.limits.maxStorageBufferRange;
      if (e.offset % alignment || e.size > limit)
        return false;
    } else if (e.image && !dynamic_cast<VulkanAllocatedImage *>(e.image))
      return false;
  }
  // Allocate all CPU storage before creating native handles. Failure leaves every
  // old slot intact, including when an array update fails halfway through.
  std::vector<VkWriteDescriptorSet> writes(resources.size());
  std::vector<VkDescriptorBufferInfo> buffers(resources.size());
  std::vector<VkDescriptorImageInfo> images(resources.size());
  auto discard = [&]() {
    for (auto image : images) {
      if (image.imageView)
        vkDestroyImageView(native, image.imageView, nullptr);
      if (image.sampler)
        vkDestroySampler(native, image.sampler, nullptr);
    }
  };
  for (size_t i = 0; i < resources.size(); ++i) {
    auto e = resources[i];
    auto binding =
        *std::find_if(layout_.bindings.begin(), layout_.bindings.end(), [e](auto b) { return b.binding == e.binding; });
    auto &write = writes[i];
    write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = Set();
    write.dstBinding = e.binding;
    write.dstArrayElement = e.array_element;
    write.descriptorCount = 1;
    write.descriptorType = Type(binding.type);
    if (binding.type <= BINDING_RW_STORAGE_BUFFER) {
      buffers[i] = {static_cast<VulkanBuffer *>(e.buffer)->NativeBuffer(), e.offset, e.size};
      write.pBufferInfo = &buffers[i];
    } else if (binding.type == BINDING_SAMPLER) {
      VkSamplerCreateInfo create{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
      create.magFilter = create.minFilter =
          e.sampler.filter == SAMPLER_FILTER_LINEAR ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
      create.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
      const VkSamplerAddressMode modes[] = {VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                            VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT};
      create.addressModeU = modes[e.sampler.address_u];
      create.addressModeV = create.addressModeW = modes[e.sampler.address_v];
      create.maxLod = VK_LOD_CLAMP_NONE;
      if (!Check(vkCreateSampler(native, &create, nullptr, &images[i].sampler), "Create sampler")) {
        discard();
        return false;
      }
      write.pImageInfo = &images[i];
    } else {
      VkImageViewCreateInfo create{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
      create.image = static_cast<VulkanImage *>(e.image)->NativeImage();
      create.viewType = VK_IMAGE_VIEW_TYPE_2D;
      create.format = NativeImageFormat(e.image->GetDesc().format);
      create.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      if (!Check(vkCreateImageView(native, &create, nullptr, &images[i].imageView), "Create binding view")) {
        discard();
        return false;
      }
      images[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
      write.pImageInfo = &images[i];
    }
  }
  vkUpdateDescriptorSets(native, uint32_t(writes.size()), writes.data(), 0, nullptr);
  for (size_t i = 0; i < resources.size(); ++i) {
    auto slot = offset + i;
    if (views_[slot])
      vkDestroyImageView(native, views_[slot], nullptr);
    if (samplers_[slot])
      vkDestroySampler(native, samplers_[slot], nullptr);
    views_[slot] = images[i].imageView;
    samplers_[slot] = images[i].sampler;
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
