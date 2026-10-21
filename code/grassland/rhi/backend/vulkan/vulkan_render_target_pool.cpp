#include "grassland/rhi/backend/vulkan/vulkan_render_target_pool.h"

#include <algorithm>

namespace grassland::rhi::backend::vulkan {
VulkanRenderTargetPool::Lease::Lease(VulkanRenderTargetPool *pool,
                                     Entry *entry,
                                     const std::vector<VulkanImage *> &images)
    : pool_(pool),
      entry_(entry) {
  images_.reserve(images.size());
  for (auto image : images)
    images_.push_back(ObjectPtr<VulkanImage>::Retain(image));
}

VulkanRenderTargetPool::Lease::~Lease() {
  pool_->ReleaseEntry(entry_);
}

const std::vector<VkImageView> &VulkanRenderTargetPool::Lease::Views() const {
  return entry_->views;
}

VkFramebuffer VulkanRenderTargetPool::Lease::Framebuffer() const {
  return entry_->framebuffer;
}

VulkanRenderTargetPool::VulkanRenderTargetPool(VkDevice device, bool dynamic_rendering)
    : device_(device),
      dynamic_rendering_(dynamic_rendering) {
}

void VulkanRenderTargetPool::DestroyEntry(Entry &entry) {
  if (entry.framebuffer)
    vkDestroyFramebuffer(device_, entry.framebuffer, nullptr);
  entry.framebuffer = {};
  entry.views.clear();
  entry.images.clear();
}

VulkanRenderTargetPool::~VulkanRenderTargetPool() {
  for (auto &entry : entries_)
    DestroyEntry(*entry);
  for (auto &pass : passes_)
    vkDestroyRenderPass(device_, pass.second, nullptr);
}

VkRenderPass VulkanRenderTargetPool::RenderPass(const std::vector<ImageFormat> &colors,
                                                ImageFormat depth,
                                                const RenderingInfo *operations) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<int> key{int(colors.size()), int(depth)};
  std::vector<VkAttachmentDescription> attachments;
  std::vector<VkAttachmentReference> references;
  constexpr VkAttachmentLoadOp loads[] = {VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_LOAD_OP_CLEAR,
                                          VK_ATTACHMENT_LOAD_OP_DONT_CARE};
  constexpr VkAttachmentStoreOp stores[] = {VK_ATTACHMENT_STORE_OP_STORE, VK_ATTACHMENT_STORE_OP_DONT_CARE};
  for (size_t i = 0; i < colors.size(); ++i) {
    auto load = operations ? operations->color_attachments[i].load_op : ATTACHMENT_LOAD_OP_LOAD;
    auto store = operations ? operations->color_attachments[i].store_op : ATTACHMENT_STORE_OP_STORE;
    key.insert(key.end(), {int(colors[i]), int(load), int(store)});
    VkAttachmentDescription a{};
    a.format = NativeImageFormat(colors[i]);
    a.samples = VK_SAMPLE_COUNT_1_BIT;
    a.loadOp = loads[load];
    a.storeOp = stores[store];
    a.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    a.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    a.initialLayout = a.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachments.push_back(a);
    references.push_back({uint32_t(i), a.initialLayout});
  }
  VkAttachmentReference depth_ref{uint32_t(colors.size()), VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
  if (depth != IMAGE_FORMAT_UNKNOWN) {
    DepthStencilAttachmentInfo defaults;
    const auto &ds = operations ? operations->depth_stencil : defaults;
    key.insert(key.end(),
               {int(ds.depth_load_op), int(ds.depth_store_op), int(ds.stencil_load_op), int(ds.stencil_store_op)});
    VkAttachmentDescription a{};
    a.format = NativeImageFormat(depth);
    a.samples = VK_SAMPLE_COUNT_1_BIT;
    a.loadOp = loads[ds.depth_load_op];
    a.storeOp = stores[ds.depth_store_op];
    a.stencilLoadOp = loads[ds.stencil_load_op];
    a.stencilStoreOp = stores[ds.stencil_store_op];
    a.initialLayout = a.finalLayout = depth_ref.layout;
    attachments.push_back(a);
  }
  auto cached = passes_.find(key);
  if (cached != passes_.end())
    return cached->second;
  // Allocate map storage before creating the native object.
  auto it = passes_.emplace(key, VK_NULL_HANDLE).first;
  VkSubpassDescription subpass{};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = uint32_t(references.size());
  subpass.pColorAttachments = references.data();
  subpass.pDepthStencilAttachment = depth == IMAGE_FORMAT_UNKNOWN ? nullptr : &depth_ref;
  VkRenderPassCreateInfo create{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
  create.attachmentCount = uint32_t(attachments.size());
  create.pAttachments = attachments.data();
  create.subpassCount = 1;
  create.pSubpasses = &subpass;
  if (!Check(vkCreateRenderPass(device_, &create, nullptr, &it->second), "Create render pass")) {
    passes_.erase(it);
    return {};
  }
  return it->second;
}

ObjectPtr<VulkanRenderTargetPool::Lease> VulkanRenderTargetPool::Acquire(const std::vector<VulkanImage *> &images,
                                                                         VkRenderPass pass) {
  try {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<VkImage> handles;
    for (auto image : images)
      handles.push_back(image->NativeImage());
    Entry *entry = nullptr;
    for (auto &candidate : entries_)
      if (candidate->images == handles) {
        entry = candidate.get();
        break;
      }
    if (!entry) {
      auto fresh = std::make_unique<Entry>();
      fresh->images = handles;
      for (auto image : images)
        fresh->views.push_back(image->AttachmentView());
      if (!dynamic_rendering_) {
        VkFramebufferCreateInfo create{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        create.renderPass = pass;
        create.attachmentCount = uint32_t(fresh->views.size());
        create.pAttachments = fresh->views.data();
        create.width = images[0]->GetDesc().width;
        create.height = images[0]->GetDesc().height;
        create.layers = 1;
        if (!Check(vkCreateFramebuffer(device_, &create, nullptr, &fresh->framebuffer), "Create framebuffer")) {
          DestroyEntry(*fresh);
          return {};
        }
      }
      entry = fresh.get();
      try {
        entries_.push_back(std::move(fresh));
      } catch (const std::bad_alloc &) {
        DestroyEntry(*entry);
        throw;
      }
    }
    auto lease = ObjectPtr<Lease>::Adopt(new Lease(this, entry, images));
    ++entry->leases;
    return lease;
  } catch (const std::bad_alloc &) {
    return {};
  }
}

void VulkanRenderTargetPool::ReleaseEntry(Entry *entry) {
  std::lock_guard<std::mutex> lock(mutex_);
  --entry->leases;
  if (!entry->leases && entry->images.empty())
    entries_.erase(std::find_if(entries_.begin(), entries_.end(), [entry](const auto &p) { return p.get() == entry; }));
}

void VulkanRenderTargetPool::Retire(VkImage image) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto it = entries_.begin(); it != entries_.end();) {
    if (std::find((*it)->images.begin(), (*it)->images.end(), image) != (*it)->images.end()) {
      DestroyEntry(**it);
      if (!(*it)->leases)
        it = entries_.erase(it);
      else
        ++it;
    } else
      ++it;
  }
}
}  // namespace grassland::rhi::backend::vulkan
