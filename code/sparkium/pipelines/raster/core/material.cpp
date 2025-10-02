#include "sparkium/pipelines/raster/core/material.h"

namespace sparkium::raster {

void Material::SetupProgram(graphics::Program *program) {
}

void Material::BindMaterialResources(graphics::CommandContext *cmd_ctx) {
}

glm::vec3 Material::Emission() const {
  return glm::vec3{0.0f};
}

}  // namespace sparkium::raster
