#pragma once
#include "grassland/graphics/backend/d3d12/d3d12_core.h"
#include "grassland/graphics/backend/d3d12/d3d12_util.h"

namespace grassland::graphics::backend {

class D3D12Shader : public Shader {
 public:
  D3D12Shader(D3D12Core *core, const CompiledShaderBlob &shader_blob);
  ~D3D12Shader() override = default;

  const std::string &EntryPoint() const override {
    return entry_point_;
  }

  d3d12::ShaderModule &ShaderModule() {
    return shader_module_;
  }

  const d3d12::ShaderModule &ShaderModule() const {
    return shader_module_;
  }

 private:
  D3D12Core *core_;
  d3d12::ShaderModule shader_module_;
  mutable std::string entry_point_;
};

}  // namespace grassland::graphics::backend
