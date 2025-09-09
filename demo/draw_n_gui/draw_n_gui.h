#pragma once
#include "long_march.h"
#include "snowberg/draw/draw.h"

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

  std::unique_ptr<snowberg::draw::Core> draw_core_;
  std::unique_ptr<snowberg::draw::Model> model_;
  std::unique_ptr<snowberg::draw::Texture> texture_;
};
