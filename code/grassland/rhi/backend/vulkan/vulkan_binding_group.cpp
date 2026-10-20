#include "grassland/rhi/backend/vulkan/vulkan_binding_group.h"

#include <algorithm>

#include "grassland/rhi/backend/vulkan/vulkan_allocated_image.h"
#include "grassland/rhi/backend/vulkan/vulkan_buffer.h"
#include "grassland/rhi/backend/vulkan/vulkan_device.h"

namespace grassland::rhi::backend::vulkan {
namespace {
VkDescriptorType Type(BindingType type) {
  const VkDescriptorType types[] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                                    VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  VK_DESCRIPTOR_TYPE_SAMPLER};
  return types[type];
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

VulkanDescriptorPool::~VulkanDescriptorPool() {
  for (auto &p : pages_)
    vkDestroyDescriptorPool(device_, p->pool, nullptr);
}

ObjectPtr<VulkanDescriptorPool::Lease> VulkanDescriptorPool::Acquire(VkDescriptorSetLayout layout) {
  try {
    std::lock_guard<std::mutex> lock(mutex_);
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate.descriptorSetCount = 1;
    allocate.pSetLayouts = &layout;
    VkDescriptorSet set{};
    Page *page = nullptr;
    for (auto &p : pages_) {
      if (p->occupied >= 64)
        continue;
      allocate.descriptorPool = p->pool;
      auto status = vkAllocateDescriptorSets(device_, &allocate, &set);
      if (status == VK_SUCCESS) {
        page = p.get();
        break;
      }
      if (status != VK_ERROR_OUT_OF_POOL_MEMORY && status != VK_ERROR_FRAGMENTED_POOL) {
        Check(status, "Allocate descriptor set");
        return {};
      }
    }
    if (!page) {
      auto fresh = std::make_unique<Page>();
      const VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1024},
                                            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1024},
                                            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1024},
                                            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1024},
                                            {VK_DESCRIPTOR_TYPE_SAMPLER, 1024}};
      VkDescriptorPoolCreateInfo create{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
      create.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
      create.maxSets = 64;
      create.poolSizeCount = 5;
      create.pPoolSizes = sizes;
      pages_.reserve(pages_.size() + 1);
      if (!Check(vkCreateDescriptorPool(device_, &create, nullptr, &fresh->pool), "Create descriptor page"))
        return {};
      page = fresh.get();
      pages_.push_back(std::move(fresh));
      allocate.descriptorPool = page->pool;
      if (!Check(vkAllocateDescriptorSets(device_, &allocate, &set), "Allocate descriptor set"))
        return {};
    }
    try {
      auto lease = ObjectPtr<Lease>::Adopt(new Lease(this, page, set));
      ++page->occupied;
      return lease;
    } catch (const std::bad_alloc &) {
      vkFreeDescriptorSets(device_, page->pool, 1, &set);
      return {};
    }
  } catch (const std::bad_alloc &) {
    return {};
  }
}

void VulkanDescriptorPool::ReleaseSet(Page *page, VkDescriptorSet set) {
  std::lock_guard<std::mutex> lock(mutex_);
  vkFreeDescriptorSets(device_, page->pool, 1, &set);
  --page->occupied;
  if (!page->occupied)
    vkResetDescriptorPool(device_, page->pool, 0);
  if (!page->occupied &&
      std::any_of(pages_.begin(), pages_.end(), [page](const auto &p) { return p.get() != page && !p->occupied; })) {
    vkDestroyDescriptorPool(device_, page->pool, nullptr);
    pages_.erase(std::find_if(pages_.begin(), pages_.end(), [page](const auto &p) { return p.get() == page; }));
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

VulkanBindingGroup::VulkanBindingGroup(VulkanDevice *device,
                                       const BindingLayoutDesc &layout,
                                       const std::vector<BindingEntry> &entries)
    : BindingGroup(device, layout, entries) {
}

VulkanBindingGroup::~VulkanBindingGroup() {
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

VkDescriptorSet VulkanBindingGroup::Set() const {
  return lease_->Set();
}

VulkanBindingGroup *VulkanBindingGroup::Create(VulkanDevice *device,
                                               const BindingLayoutDesc &layout,
                                               const std::vector<BindingEntry> &entries) {
  try {
    if (!Validate(device, layout, entries) || !ValidatePipelineBindings(device->NativePhysicalDevice(), {{layout}}))
      return nullptr;
    for (const auto &entry : entries)
      if (entry.image && !dynamic_cast<VulkanAllocatedImage *>(entry.image))
        return nullptr;
    auto group = std::unique_ptr<VulkanBindingGroup>(new VulkanBindingGroup(device, layout, entries));
    auto native = device->NativeDevice();
    group->native_layout_ = CreateBindingLayout(native, layout);
    if (!group->native_layout_)
      return nullptr;
    group->lease_ = device->DescriptorPool().Acquire(group->native_layout_);
    if (!group->lease_)
      return nullptr;
    group->views_.resize(entries.size());
    group->samplers_.resize(entries.size());
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(device->NativePhysicalDevice(), &properties);
    for (size_t i = 0; i < group->entries_.size(); ++i) {
      auto e = group->entries_[i];
      auto binding =
          *std::find_if(layout.bindings.begin(), layout.bindings.end(), [e](auto b) { return b.binding == e.binding; });
      VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      write.dstSet = group->Set();
      write.dstBinding = e.binding;
      write.dstArrayElement = e.array_element;
      write.descriptorCount = 1;
      write.descriptorType = Type(binding.type);
      VkDescriptorBufferInfo buffer{};
      VkDescriptorImageInfo image{};
      if (binding.type <= BINDING_RW_STORAGE_BUFFER) {
        bool uniform = binding.type == BINDING_UNIFORM_BUFFER;
        auto alignment = uniform ? properties.limits.minUniformBufferOffsetAlignment
                                 : properties.limits.minStorageBufferOffsetAlignment;
        auto limit = uniform ? properties.limits.maxUniformBufferRange : properties.limits.maxStorageBufferRange;
        if (e.offset % alignment || e.size > limit)
          return nullptr;
        buffer = {static_cast<VulkanBuffer *>(e.buffer)->NativeBuffer(), e.offset, e.size};
        write.pBufferInfo = &buffer;
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
        if (!Check(vkCreateSampler(native, &create, nullptr, &group->samplers_[i]), "Create sampler"))
          return nullptr;
        image.sampler = group->samplers_[i];
        write.pImageInfo = &image;
      } else {
        VkImageViewCreateInfo create{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        create.image = static_cast<VulkanImage *>(e.image)->NativeImage();
        create.viewType = VK_IMAGE_VIEW_TYPE_2D;
        create.format = ColorFormat(e.image->GetDesc().format);
        create.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (!Check(vkCreateImageView(native, &create, nullptr, &group->views_[i]), "Create binding view"))
          return nullptr;
        image.imageView = group->views_[i];
        image.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        write.pImageInfo = &image;
      }
      vkUpdateDescriptorSets(native, 1, &write, 0, nullptr);
    }
    return group.release();
  } catch (const std::bad_alloc &) {
    return nullptr;
  }
}

bool ValidatePipelineBindings(VkPhysicalDevice device, const PipelineLayoutDesc &layout) {
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(device, &properties);
  const auto &limits = properties.limits;
  if (layout.groups.size() > limits.maxBoundDescriptorSets)
    return false;
  uint32_t totals[5]{}, stages[2][5]{};
  for (const auto &group : layout.groups)
    for (auto b : group.bindings) {
      int kind = b.type == BINDING_UNIFORM_BUFFER      ? 0
                 : b.type <= BINDING_RW_STORAGE_BUFFER ? 1
                 : b.type == BINDING_TEXTURE2D         ? 2
                 : b.type == BINDING_RW_TEXTURE2D      ? 3
                                                       : 4;
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
  return {pages_.size(),
          size_t(std::count_if(pages_.begin(), pages_.end(), [](const auto &p) { return p->occupied != 0; }))};
}

}  // namespace grassland::rhi::backend::vulkan
