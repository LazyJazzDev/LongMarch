#pragma once
#include "grassland/graphics/graphics.h"
#include "long_march.h"

class Application {
 public:
  Application(grassland::graphics::BackendAPI api =
                  grassland::graphics::BACKEND_API_VULKAN);

  ~Application();

 private:
  std::shared_ptr<grassland::graphics::Core> core_;
};
