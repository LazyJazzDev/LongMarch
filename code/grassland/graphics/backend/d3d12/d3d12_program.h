#pragma once
#include "grassland/graphics/backend/d3d12/d3d12_core.h"
#include "grassland/graphics/backend/d3d12/d3d12_util.h"

namespace grassland::graphics::backend {
class D3D12Shader : public Shader {
 public:
  D3D12Shader(const void *data, size_t size, D3D12Core *core);
  ~D3D12Shader() override = default;

 private:
  D3D12Core *core_;
  Microsoft::WRL::ComPtr<ID3DBlob> shader_blob_;
};
}  // namespace grassland::graphics::backend
