#include "grassland/graphics/backend/d3d12/d3d12_shader.h"

#include "grassland/util/string_convert.h"

namespace grassland::graphics::backend {

D3D12Shader::D3D12Shader(D3D12Core *core, const CompiledShaderBlob &blob) : core_(core), shader_module_(blob) {
}

const std::string &D3D12Shader::EntryPoint() const {
  return WStringToString(shader_module_.EntryPoint());
}

}  // namespace grassland::graphics::backend
