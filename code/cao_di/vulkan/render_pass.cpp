#include "cao_di/vulkan/render_pass.h"

#include "cao_di/vulkan/framebuffer.h"

namespace CD::vulkan {

VkSubpassDescription SubpassSettings::Description() const {
  VkSubpassDescription description{};
  description.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  description.inputAttachmentCount = input_attachment_references.size();
  description.pInputAttachments = input_attachment_references.data();
  description.colorAttachmentCount = color_attachment_references.size();
  description.pColorAttachments = color_attachment_references.data();
  if (depth_attachment_reference.has_value()) {
    description.pDepthStencilAttachment = &depth_attachment_reference.value();
  }
  if (!resolve_attachment_references.empty()) {
    description.pResolveAttachments = resolve_attachment_references.data();
  }
  description.preserveAttachmentCount = preserve_attachment_references.size();
  if (description.preserveAttachmentCount) {
    description.pPreserveAttachments = preserve_attachment_references.data();
  }
  return description;
}

SubpassSettings::SubpassSettings(const std::vector<VkAttachmentReference> &color_attachment_references,
                                 const std::optional<VkAttachmentReference> &depth_attachment_reference,
                                 const std::vector<VkAttachmentReference> &resolve_attachment_references)
    : color_attachment_references(color_attachment_references),
      depth_attachment_reference(depth_attachment_reference),
      resolve_attachment_references(resolve_attachment_references) {
}

SubpassSettings::SubpassSettings(const std::vector<VkAttachmentReference> &input_attachment_references,
                                 const std::vector<VkAttachmentReference> &color_attachment_references,
                                 const std::optional<VkAttachmentReference> &depth_attachment_reference,
                                 const std::vector<VkAttachmentReference> &resolve_attachment_references,
                                 const std::vector<uint32_t> &preserve_attachment_references)
    : input_attachment_references(input_attachment_references),
      color_attachment_references(color_attachment_references),
      depth_attachment_reference(depth_attachment_reference),
      resolve_attachment_references(resolve_attachment_references),
      preserve_attachment_references(preserve_attachment_references) {
}

RenderPass::RenderPass(const struct Device *device,
                       std::vector<VkAttachmentDescription> attachment_descriptions,
                       std::vector<struct SubpassSettings> subpass_settings,
                       VkRenderPass render_pass)
    : device_(device),
      attachment_descriptions_(std::move(attachment_descriptions)),
      subpass_settings_(std::move(subpass_settings)),
      render_pass_(render_pass) {
}

RenderPass::~RenderPass() {
  vkDestroyRenderPass(device_->Handle(), render_pass_, nullptr);
}

VkResult RenderPass::CreateFramebuffer(const std::vector<VkImageView> &image_views,
                                       VkExtent2D extent,
                                       double_ptr<Framebuffer> pp_framebuffer) const {
  if (!pp_framebuffer) {
    SetErrorMessage("pp_framebuffer is nullptr");
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  VkFramebufferCreateInfo framebuffer_create_info{};
  framebuffer_create_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  framebuffer_create_info.renderPass = render_pass_;
  framebuffer_create_info.attachmentCount = static_cast<uint32_t>(image_views.size());
  framebuffer_create_info.pAttachments = image_views.data();
  framebuffer_create_info.width = extent.width;
  framebuffer_create_info.height = extent.height;
  framebuffer_create_info.layers = 1;

  VkFramebuffer framebuffer;
  RETURN_IF_FAILED_VK(vkCreateFramebuffer(device_->Handle(), &framebuffer_create_info, nullptr, &framebuffer),
                      "failed to create framebuffer!");

  pp_framebuffer.construct(this, extent, framebuffer);

  return VK_SUCCESS;
}
}  // namespace CD::vulkan
