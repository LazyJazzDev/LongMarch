#pragma once
#include "grassland/graphics/graphics.h"
#include "long_march.h"

class Application {
 public:
  Application();
  ~Application() = default;

 private:
  std::shared_ptr<grassland::graphics::Core> core_;
};
