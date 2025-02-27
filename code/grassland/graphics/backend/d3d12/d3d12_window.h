#pragma once
#include "grassland/graphics/backend/d3d12/d3d12_core.h"
#include "grassland/graphics/backend/d3d12/d3d12_util.h"

namespace grassland::graphics::backend {

class D3D12Window : public Window {
 public:
  D3D12Window(D3D12Core *core,
              int width,
              int height,
              const std::string &title,
              bool fullscreen,
              bool resizable,
              bool enable_hdr);

  virtual void CloseWindow() override;

  d3d12::SwapChain *SwapChain() const;

  ID3D12Resource *CurrentBackBuffer() const;

 private:
  D3D12Core *core_;
  std::unique_ptr<d3d12::SwapChain> swap_chain_;
  uint32_t swap_chain_recreate_event_id_;
};

}  // namespace grassland::graphics::backend
