#include "sparkium/pipelines/raster/material/material_light.h"

#include "sparkium/pipelines/raster/core/core.h"

namespace sparkium::raster {

MaterialLight::MaterialLight(sparkium::MaterialLight &material)
    : material_(material), Material(DedicatedCast(material.GetCore())) {
}

graphics::Shader *MaterialLight::PixelShader() {
  return nullptr;
}

graphics::Buffer *MaterialLight::Buffer() {
  return nullptr;
}

void MaterialLight::Sync() {
}

}  // namespace sparkium::raster
