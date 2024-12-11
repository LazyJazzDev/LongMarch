#pragma once
#include "grassland/graphics/backend/vulkan/vulkan_core.h"
#include "grassland/graphics/backend/vulkan/vulkan_util.h"

namespace grassland::graphics::backend {

class VulkanWindow : public Window {
 public:
  VulkanWindow(int width,
               int height,
               const std::string &title,
               VulkanCore *core);

  virtual void CloseWindow() override;

 private:
  VulkanCore *core_;
  std::unique_ptr<vulkan::Surface> surface_;
  std::unique_ptr<vulkan::Swapchain> swap_chain_;
};

}  // namespace grassland::graphics::backend
