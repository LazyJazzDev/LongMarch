#include "grassland/graphics/backend/vulkan/helper/framebuffer.h"

namespace grassland::graphics::backend::vulkan {

Framebuffer::Framebuffer(const struct RenderPass *render_pass, VkExtent2D extent, VkFramebuffer framebuffer)
    : render_pass_(render_pass),
      extent_(extent),
      framebuffer_(framebuffer) {
}

Framebuffer::~Framebuffer() {
  vkDestroyFramebuffer(render_pass_->Device()->Handle(), framebuffer_, nullptr);
}
}  // namespace grassland::graphics::backend::vulkan
