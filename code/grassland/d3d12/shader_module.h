#pragma once
#include "grassland/d3d12/device.h"

namespace grassland::d3d12 {

class ShaderModule {
 public:
  ShaderModule(const CompiledShaderBlob &shader_blob);

  D3D12_SHADER_BYTECODE Handle() const {
    return {shader_code_.data(), shader_code_.size()};
  }

  const std::wstring &EntryPoint() const {
    return entry_point_;
  }

 private:
  std::vector<uint8_t> shader_code_;
  std::wstring entry_point_;
};

ComPtr<ID3DBlob> CompileShaderLegacy(const std::string &source_code,
                                     const std::string &entry_point,
                                     const std::string &target);

CompiledShaderBlob CompileShader(const std::string &source_code,
                                 const std::string &entry_point,
                                 const std::string &target);

}  // namespace grassland::d3d12
