#pragma once

#include "sparkium/core/material.h"

namespace sparkium {

// A material whose surface parameters are evaluated by Slang generated from a
// JSON shader-node graph. Texture indices are assigned when the scene updates.
class MaterialShaderGraph : public Material {
 public:
  MaterialShaderGraph(Core *core,
                      const CodeLines &graph_code,
                      const std::vector<graphics::Image *> &textures,
                      const glm::vec3 &emission_hint = {});

  CodeLines graph_code;
  std::vector<graphics::Image *> textures;
  glm::vec3 emission_hint{};
};

}  // namespace sparkium
