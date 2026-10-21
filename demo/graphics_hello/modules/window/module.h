#pragma once
#include "../../module.h"

namespace graphics_hello::window {
class ModuleWindow final : public Module {
 public:
  ModuleWindow(grassland::graphics::BackendAPI api, bool headless = false);
  void OnInit() override;
  void OnClose() override;
  void OnUpdate() override;
  void OnRender() override;
  grassland::graphics::Core *GetCore() const override;
  grassland::graphics::Window *GetWindow() const override;
  bool IsAlive() const override;

 private:
  std::shared_ptr<grassland::graphics::Core> core_;
  std::unique_ptr<grassland::graphics::Window> window_;
  std::array<std::unique_ptr<grassland::graphics::Image>, 3> color_images_;
  bool alive_{};
};
}  // namespace graphics_hello::window
