#pragma once
#include "grassland/d3d12/d3d12util.h"
#include "grassland/util/double_ptr.h"

namespace grassland::d3d12 {
class DXGIFactory {
 public:
  explicit DXGIFactory(IDXGIFactory4 *factory) : factory_(factory) {
  }

 private:
  ComPtr<IDXGIFactory4> factory_;
};

HRESULT CreateDXGIFactory(double_ptr<DXGIFactory> pp_factory);

}  // namespace grassland::d3d12
