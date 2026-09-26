#include "grassland/d3d12/shader_module.h"

#include "grassland/graphics/program.h"

namespace grassland::d3d12 {
ShaderModule::ShaderModule(const CompiledShaderBlob &shader_blob)
    : shader_code_(shader_blob.data),
      entry_point_(StringToWString(shader_blob.entry_point)) {
}

CompiledShaderBlob CompileShader(const std::string &source_code,
                                 const std::string &entry_point,
                                 const std::string &target) {
  return graphics::CompileShader(source_code, entry_point, target);
}

}  // namespace grassland::d3d12
