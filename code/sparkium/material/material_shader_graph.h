#pragma once

#include "sparkium/core/material.h"
#include "sparkium/core/shader_graph_program.h"

namespace sparkium {

// A material whose surface parameters are evaluated by HLSL generated from a
// JSON shader-node graph. Texture indices are assigned when the scene updates.
// `program` describes the same graph as a backend-neutral instruction list for
// the native CPU/CUDA backends, which have no HLSL compiler.
class MaterialShaderGraph : public Material {
 public:
  MaterialShaderGraph(Core *core, const CodeLines &graph_code,
                      const std::vector<graphics::Image *> &textures,
                      const glm::vec3 &emission_hint = {}, const ShaderGraphProgram &program = {});

  CodeLines graph_code;
  std::vector<graphics::Image *> textures;
  glm::vec3 emission_hint{};
  ShaderGraphProgram program;
};

}  // namespace sparkium
