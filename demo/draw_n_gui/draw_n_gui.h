#pragma once
#include "long_march.h"
#include "snow_mount/draw/draw.h"

class DrawNGUI {
 public:
  DrawNGUI(grassland::graphics::BackendAPI api);
  ~DrawNGUI();
  void Run();

 private:
  void OnInit();
  void OnClose();
  void OnUpdate();
  void OnRender();

  std::unique_ptr<grassland::graphics::Core> core_;
  std::unique_ptr<grassland::graphics::Image> color_image_;
  std::unique_ptr<grassland::graphics::Window> window_;

  std::unique_ptr<snow_mount::draw::Core> draw_core_;
  std::unique_ptr<snow_mount::draw::Model> model_;
  std::unique_ptr<snow_mount::draw::Texture> texture_;
};
