#pragma once
#include "long_march.h"
#include "snow_mount/draw/draw.h"

class DrawNGUI {
 public:
  DrawNGUI(CD::graphics::BackendAPI api);
  ~DrawNGUI();
  void Run();

 private:
  void OnInit();
  void OnClose();
  void OnUpdate();
  void OnRender();

  std::unique_ptr<CD::graphics::Core> core_;
  std::unique_ptr<CD::graphics::Image> color_image_;
  std::unique_ptr<CD::graphics::Window> window_;

  std::unique_ptr<XS::draw::Core> draw_core_;
  std::unique_ptr<XS::draw::Model> model_;
  std::unique_ptr<XS::draw::Texture> texture_;
};
