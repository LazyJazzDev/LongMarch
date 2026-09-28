#pragma once
#include "grassland/graphics/backend/d3d12/helper/device.h"

namespace grassland::graphics::backend::d3d12 {

class RootSignature {
 public:
  RootSignature(const ComPtr<ID3D12RootSignature> &root_signature);

  ID3D12RootSignature *Handle() const {
    return root_signature_.Get();
  }

 private:
  ComPtr<ID3D12RootSignature> root_signature_;
};

}  // namespace grassland::graphics::backend::d3d12
