#include "grassland/graphics/backend/d3d12/helper/shader_module.h"

#include "grassland/graphics/program.h"

namespace grassland::graphics::backend::d3d12 {
CompiledShaderBlob CompileShader(const std::string &source_code,
                                 const std::string &entry_point,
                                 const std::string &target) {
  return graphics::CompileShader(source_code, entry_point, target);
}

}  // namespace grassland::graphics::backend::d3d12
