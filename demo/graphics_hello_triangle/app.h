#pragma once
#include "grassland/graphics/graphics.h"
#include "long_march.h"

class Application {
 public:
  Application(grassland::graphics::BackendAPI api =
                  grassland::graphics::BACKEND_API_VULKAN);

  ~Application();

  void OnInit();
  void OnClose();
  void OnUpdate();
  void OnRender();

  bool IsAlive() const {
    return alive_;
  }

 private:
  std::shared_ptr<grassland::graphics::Core> core_;
  std::unique_ptr<grassland::graphics::Window> window_;
  bool alive_{true};
};
