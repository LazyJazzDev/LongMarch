#pragma once

#include <optional>

#include "grassland/vulkan/device.h"

namespace CD::vulkan {
struct SubpassSettings {
  std::vector<VkAttachmentReference> input_attachment_references{};
  std::vector<VkAttachmentReference> color_attachment_references{};
  std::optional<VkAttachmentReference> depth_attachment_reference{};
  std::vector<VkAttachmentReference> resolve_attachment_references{};
  std::vector<uint32_t> preserve_attachment_references{};

  SubpassSettings() = default;

  SubpassSettings(const std::vector<VkAttachmentReference> &color_attachment_references,
                  const std::optional<VkAttachmentReference> &depth_attachment_reference = std::nullopt,
                  const std::vector<VkAttachmentReference> &resolve_attachment_references = {});

  SubpassSettings(const std::vector<VkAttachmentReference> &input_attachment_references,
                  const std::vector<VkAttachmentReference> &color_attachment_references,
                  const std::optional<VkAttachmentReference> &depth_attachment_reference = std::nullopt,
                  const std::vector<VkAttachmentReference> &resolve_attachment_references = {},
                  const std::vector<uint32_t> &preserve_attachment_references = {});

  VkSubpassDescription Description() const;

  const std::vector<VkAttachmentReference> &InputAttachmentReferences() const {
    return input_attachment_references;
  }

  const std::vector<VkAttachmentReference> &ColorAttachmentReferences() const {
    return color_attachment_references;
  }

  const std::optional<VkAttachmentReference> &DepthAttachmentReference() const {
    return depth_attachment_reference;
  }

  const std::vector<VkAttachmentReference> &ResolveAttachmentReferences() const {
    return resolve_attachment_references;
  }

  const std::vector<uint32_t> &PreserveAttachmentReferences() const {
    return preserve_attachment_references;
  }
};

class RenderPass {
 public:
  RenderPass(const class Device *device,
             std::vector<VkAttachmentDescription> attachment_descriptions,
             std::vector<struct SubpassSettings> subpass_settings,
             VkRenderPass render_pass);

  ~RenderPass();

  const class Device *Device() const {
    return device_;
  }

  VkRenderPass Handle() const {
    return render_pass_;
  }

  const std::vector<VkAttachmentDescription> &AttachmentDescriptions() const {
    return attachment_descriptions_;
  }

  const std::vector<struct SubpassSettings> &SubpassSettings() const {
    return subpass_settings_;
  }

  VkResult CreateFramebuffer(const std::vector<VkImageView> &image_views,
                             VkExtent2D extent,
                             double_ptr<Framebuffer> pp_framebuffer) const;

 private:
  const class Device *device_{};
  std::vector<VkAttachmentDescription> attachment_descriptions_{};
  std::vector<struct SubpassSettings> subpass_settings_{};
  VkRenderPass render_pass_{};
};
}  // namespace CD::vulkan
