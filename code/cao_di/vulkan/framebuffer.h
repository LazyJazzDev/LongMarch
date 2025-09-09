#pragma once

#include "cao_di/vulkan/device.h"
#include "cao_di/vulkan/render_pass.h"

namespace CD::vulkan {
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
}  // namespace CD::vulkan
