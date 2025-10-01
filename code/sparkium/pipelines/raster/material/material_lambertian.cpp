#include "sparkium/pipelines/raster/material/material_lambertian.h"

#include "sparkium/pipelines/raster/core/core.h"

namespace sparkium::raster {

MaterialLambertian::MaterialLambertian(sparkium::MaterialLambertian &material)
    : material_(material), Material(DedicatedCast(material.GetCore())) {
}

graphics::Shader *MaterialLambertian::PixelShader() {
  return nullptr;
}

}  // namespace sparkium::raster
