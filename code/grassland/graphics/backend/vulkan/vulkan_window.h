#pragma once
#include "grassland/graphics/backend/vulkan/vulkan_util.h"

namespace grassland::graphics::backend {

class VulkanWindow : public Window {
 public:
  VulkanWindow(int width, int height, const std::string &title);

  virtual void CloseWindow() override;

 private:
};

}  // namespace grassland::graphics::backend
