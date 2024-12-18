#pragma once
#include "grassland/d3d12/device.h"

namespace grassland::d3d12 {

class ShaderModule {
 public:
  ShaderModule(const std::vector<uint8_t> &shader_code);
  ;

  D3D12_SHADER_BYTECODE Handle() const {
    return {shader_code_.data(), shader_code_.size()};
  }

 private:
  std::vector<uint8_t> shader_code_;
};

ComPtr<ID3DBlob> CompileShaderLegacy(const std::string &source_code,
                                     const std::string &entry_point,
                                     const std::string &target);

ComPtr<ID3DBlob> CompileShader(const std::string &source_code,
                               const std::string &entry_point,
                               const std::string &target);

}  // namespace grassland::d3d12
