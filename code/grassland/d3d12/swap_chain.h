#pragma once
#include "grassland/d3d12/dxgi_factory.h"

namespace grassland::d3d12 {
class SwapChain {
 public:
  SwapChain(const ComPtr<IDXGISwapChain3> &swap_chain);

 private:
  ComPtr<IDXGISwapChain3> swap_chain_;
};
}  // namespace grassland::d3d12
