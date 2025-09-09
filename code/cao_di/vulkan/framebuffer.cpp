#include "cao_di/vulkan/framebuffer.h"

namespace CD::vulkan {

Framebuffer::Framebuffer(const struct RenderPass *render_pass, VkExtent2D extent, VkFramebuffer framebuffer)
    : render_pass_(render_pass), extent_(extent), framebuffer_(framebuffer) {
}

Framebuffer::~Framebuffer() {
  vkDestroyFramebuffer(render_pass_->Device()->Handle(), framebuffer_, nullptr);
}
}  // namespace CD::vulkan
