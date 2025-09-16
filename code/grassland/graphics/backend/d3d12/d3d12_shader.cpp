#include "grassland/graphics/backend/d3d12/d3d12_shader.h"

#include "grassland/util/string_convert.h"

namespace grassland::graphics::backend {

D3D12Shader::D3D12Shader(D3D12Core *core, const CompiledShaderBlob &blob) : core_(core), shader_module_(blob) {
}

const std::string &D3D12Shader::EntryPoint() const {
  if (entry_point_.empty()) {
    entry_point_ = WStringToString(shader_module_.EntryPoint());
  }
  return entry_point_;
}

}  // namespace grassland::graphics::backend
