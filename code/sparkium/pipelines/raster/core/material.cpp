#include "sparkium/pipelines/raster/core/material.h"

namespace sparkium::raster {

void Material::SetupProgram(rhi::ProgramDesc &program) {
}

bool Material::BindMaterialResources(rhi::IBindingGroup *group) {
  return true;
}

glm::vec3 Material::Emission() const {
  return glm::vec3{0.0f};
}

}  // namespace sparkium::raster
