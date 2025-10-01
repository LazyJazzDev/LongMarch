#include "sparkium/pipelines/raster/material/material_specular.h"

#include "sparkium/pipelines/raster/core/core.h"

namespace sparkium::raster {

MaterialSpecular::MaterialSpecular(sparkium::MaterialSpecular &material)
    : material_(material), Material(DedicatedCast(material.GetCore())) {
}

graphics::Shader *MaterialSpecular::PixelShader() {
  return nullptr;
}

}  // namespace sparkium::raster
