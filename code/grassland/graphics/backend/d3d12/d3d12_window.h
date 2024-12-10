#pragma once
#include "grassland/graphics/backend/d3d12/d3d12_util.h"

namespace grassland::graphics::backend {

class D3D12Window : public Window {
 public:
  D3D12Window(int width, int height, const std::string &title);

  virtual void CloseWindow() override;

 private:
};

}  // namespace grassland::graphics::backend
