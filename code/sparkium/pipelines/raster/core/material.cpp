#include "sparkium/pipelines/raster/core/material.h"

namespace sparkium::raster {

glm::vec3 Material::Emission() const {
  return glm::vec3{0.0f};
}

}  // namespace sparkium::raster
