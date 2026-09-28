#pragma once
#include "grassland/graphics/backend/d3d12/helper/d3d12util.h"

namespace grassland::graphics::backend::d3d12 {

struct HitGroup {
  const CompiledShaderBlob *closest_hit_shader{nullptr};
  const CompiledShaderBlob *any_hit_shader{nullptr};
  const CompiledShaderBlob *intersection_shader{nullptr};
  bool procedure{false};
};

CompiledShaderBlob CompileShader(const std::string &source_code,
                                 const std::string &entry_point,
                                 const std::string &target);

}  // namespace grassland::graphics::backend::d3d12
