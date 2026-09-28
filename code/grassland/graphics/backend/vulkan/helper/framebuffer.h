#pragma once

#include "grassland/graphics/backend/vulkan/helper/device.h"
#include "grassland/graphics/backend/vulkan/helper/render_pass.h"

namespace grassland::graphics::backend::vulkan {
class Framebuffer {
 public:
  Framebuffer(const class RenderPass *render_pass, VkExtent2D extent, VkFramebuffer framebuffer);

  ~Framebuffer();

  VkFramebuffer Handle() const {
    return framebuffer_;
  }

  const class RenderPass *RenderPass() const {
    return render_pass_;
  }

  VkExtent2D Extent() const {
    return extent_;
  }

 private:
  const class RenderPass *render_pass_;
  VkExtent2D extent_{};
  VkFramebuffer framebuffer_{};
};
}  // namespace grassland::graphics::backend::vulkan
