#include "sparkium/pipelines/raster/core/material.h"

namespace sparkium::raster {

void Material::SetupProgram(rhi::ProgramDesc &program) {
}

void Material::BindMaterialResources(rhi::IBindingGroup *group) {
}

glm::vec3 Material::Emission() const {
  return glm::vec3{0.0f};
}

}  // namespace sparkium::raster
