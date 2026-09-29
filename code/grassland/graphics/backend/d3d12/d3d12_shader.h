#pragma once
#include "grassland/graphics/backend/d3d12/d3d12_core.h"
#include "grassland/graphics/backend/d3d12/d3d12_util.h"

namespace grassland::graphics::backend {

class D3D12Shader : public Shader {
 public:
  D3D12Shader(D3D12Core *core, const CompiledShaderBlob &shader_blob);
  ~D3D12Shader() override = default;

  std::string EntryPoint() const override;

  D3D12_SHADER_BYTECODE Bytecode() const {
    return {shader_blob_.data.data(), shader_blob_.data.size()};
  }

  const CompiledShaderBlob &CompiledBlob() const {
    return shader_blob_;
  }

 private:
  D3D12Core *core_;
  CompiledShaderBlob shader_blob_;
};

}  // namespace grassland::graphics::backend
