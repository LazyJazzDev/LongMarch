#include "sparkium/pipelines/raster/material/material_principled.h"

#include "sparkium/pipelines/raster/core/core.h"

namespace sparkium::raster {

MaterialPrincipled::MaterialPrincipled(sparkium::MaterialPrincipled &material)
    : material_(material), Material(DedicatedCast(material.GetCore())) {
}

graphics::Shader *MaterialPrincipled::PixelShader() {
  return nullptr;
}

graphics::Buffer *MaterialPrincipled::Buffer() {
  return nullptr;
}

void MaterialPrincipled::Sync() {
}

}  // namespace sparkium::raster
