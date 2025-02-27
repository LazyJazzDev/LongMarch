#pragma once
#include "grassland/graphics/backend/vulkan/vulkan_core.h"
#include "grassland/graphics/backend/vulkan/vulkan_util.h"

namespace grassland::graphics::backend {

class VulkanWindow : public Window {
 public:
  VulkanWindow(VulkanCore *core,
               int width,
               int height,
               const std::string &title,
               bool fullscreen,
               bool resizable,
               bool enable_hdr);

  virtual void CloseWindow() override;

  vulkan::Swapchain *SwapChain() const;

  vulkan::Semaphore *RenderFinishSemaphore() const;

  vulkan::Semaphore *ImageAvailableSemaphore() const;

  uint32_t AcquireNextImage();

  void Rebuild();

  void Present();

  uint32_t CurrentImageIndex() const {
    return image_index_;
  }

  VkImage CurrentImage() const {
    return swap_chain_->Image(image_index_);
  }

 private:
  VkQueue present_queue_;
  VulkanCore *core_;
  std::unique_ptr<vulkan::Surface> surface_;
  std::unique_ptr<vulkan::Swapchain> swap_chain_;
  std::vector<std::unique_ptr<vulkan::Semaphore>> render_finish_semaphores_;
  std::vector<std::unique_ptr<vulkan::Semaphore>> image_available_semaphores_;
  uint32_t image_index_;
};

}  // namespace grassland::graphics::backend
