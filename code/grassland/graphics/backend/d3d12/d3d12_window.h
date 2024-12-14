#pragma once
#include "grassland/graphics/backend/d3d12/d3d12_core.h"
#include "grassland/graphics/backend/d3d12/d3d12_util.h"

namespace grassland::graphics::backend {

class D3D12Window : public Window {
 public:
  D3D12Window(D3D12Core *core, int width, int height, const std::string &title);

  virtual void CloseWindow() override;

 private:
  D3D12Core *core_;
  std::unique_ptr<d3d12::SwapChain> swap_chain_;
};

}  // namespace grassland::graphics::backend
