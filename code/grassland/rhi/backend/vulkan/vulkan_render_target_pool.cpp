#include "grassland/rhi/backend/vulkan/vulkan_render_target_pool.h"

namespace grassland::rhi::backend::vulkan {
VkFormat ColorFormat(ImageFormat format) {
  return format == IMAGE_FORMAT_RGBA8_UNORM ? VK_FORMAT_R8G8B8A8_UNORM : VK_FORMAT_B8G8R8A8_UNORM;
}

VulkanRenderTargetPool::VulkanRenderTargetPool(VkDevice device, bool dynamic_rendering)
    : device_(device),
      dynamic_rendering_(dynamic_rendering) {
}

VulkanRenderTargetPool::~VulkanRenderTargetPool() {
  for (auto &entry : entries_) {
    if (entry.framebuffer)
      vkDestroyFramebuffer(device_, entry.framebuffer, nullptr);
    if (entry.view)
      vkDestroyImageView(device_, entry.view, nullptr);
  }
  for (auto pass : passes_)
    if (pass)
      vkDestroyRenderPass(device_, pass, nullptr);
}

VkRenderPass VulkanRenderTargetPool::CreateRenderPass(ImageFormat format,
                                                      AttachmentLoadOp load_op,
                                                      AttachmentStoreOp store_op) {
  auto &pass = passes_[(format == IMAGE_FORMAT_RGBA8_UNORM ? 0 : 6) + static_cast<int>(load_op) * 2 +
                       static_cast<int>(store_op)];
  if (pass)
    return pass;
  VkAttachmentDescription attachment{};
  attachment.format = ColorFormat(format);
  attachment.samples = VK_SAMPLE_COUNT_1_BIT;
  constexpr VkAttachmentLoadOp loads[] = {VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_LOAD_OP_CLEAR,
                                          VK_ATTACHMENT_LOAD_OP_DONT_CARE};
  attachment.loadOp = loads[load_op];
  constexpr VkAttachmentStoreOp stores[] = {VK_ATTACHMENT_STORE_OP_STORE, VK_ATTACHMENT_STORE_OP_DONT_CARE};
  attachment.storeOp = stores[store_op];
  attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  attachment.initialLayout = attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  VkAttachmentReference color{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
  VkSubpassDescription subpass{};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &color;
  VkRenderPassCreateInfo create{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
  create.attachmentCount = 1;
  create.pAttachments = &attachment;
  create.subpassCount = 1;
  create.pSubpasses = &subpass;
  if (!Check(vkCreateRenderPass(device_, &create, nullptr, &pass), "Create render pass"))
    return {};
  return pass;
}

VkRenderPass VulkanRenderTargetPool::RenderPass(ImageFormat format,
                                                AttachmentLoadOp load_op,
                                                AttachmentStoreOp store_op) {
  std::lock_guard<std::mutex> lock(mutex_);
  return CreateRenderPass(format, load_op, store_op);
}

bool VulkanRenderTargetPool::Acquire(VkImage image,
                                     const ImageDesc &desc,
                                     uint64_t &lease,
                                     VkImageView &image_view,
                                     VkFramebuffer &framebuffer) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (size_t i = 0; i < entries_.size(); ++i)
    if (entries_[i].image == image) {
      ++entries_[i].leases;
      lease = i;
      image_view = entries_[i].view;
      framebuffer = entries_[i].framebuffer;
      return true;
    }
  VkRenderPass pass{};
  if (!dynamic_rendering_) {
    pass = CreateRenderPass(desc.format);
    if (!pass)
      return false;
  }
  size_t slot = 0;
  for (; slot < entries_.size(); ++slot)
    if (!entries_[slot].image && !entries_[slot].leases)
      break;
  if (slot == entries_.size())
    entries_.emplace_back();
  auto &entry = entries_[slot];
  VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  view.image = image;
  view.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view.format = ColorFormat(desc.format);
  view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  if (!Check(vkCreateImageView(device_, &view, nullptr, &entry.view), "Create render-target view"))
    return false;
  if (!dynamic_rendering_) {
    VkFramebufferCreateInfo create{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    create.renderPass = pass;
    create.attachmentCount = 1;
    create.pAttachments = &entry.view;
    create.width = desc.width;
    create.height = desc.height;
    create.layers = 1;
    if (!Check(vkCreateFramebuffer(device_, &create, nullptr, &entry.framebuffer), "Create framebuffer")) {
      vkDestroyImageView(device_, entry.view, nullptr);
      entry.view = {};
      return false;
    }
  }
  entry.image = image;
  entry.leases = 1;
  lease = slot;
  image_view = entry.view;
  framebuffer = entry.framebuffer;
  return true;
}

void VulkanRenderTargetPool::Release(uint64_t lease) {
  std::lock_guard<std::mutex> lock(mutex_);
  --entries_[lease].leases;
}

void VulkanRenderTargetPool::Retire(VkImage image) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto &entry : entries_)
    if (entry.image == image) {
      if (entry.framebuffer)
        vkDestroyFramebuffer(device_, entry.framebuffer, nullptr);
      vkDestroyImageView(device_, entry.view, nullptr);
      entry.image = {};
      entry.view = {};
      entry.framebuffer = {};
      // A completed context may still own its CPU lease; do not recycle its slot yet.
    }
}
}  // namespace grassland::rhi::backend::vulkan
