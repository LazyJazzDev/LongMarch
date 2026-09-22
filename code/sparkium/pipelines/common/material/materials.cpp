#include "sparkium/pipelines/common/material/materials.h"

namespace sparkium::render_shared {

Material *DedicatedCast(sparkium::Material *material) {
  DEDICATED_CAST(material, sparkium::MaterialLambertian, MaterialLambertian);
  DEDICATED_CAST(material, sparkium::MaterialLight, MaterialLight);
  DEDICATED_CAST(material, sparkium::MaterialSpecular, MaterialSpecular);
  DEDICATED_CAST(material, sparkium::MaterialPrincipled, MaterialPrincipled);
  DEDICATED_CAST(material, sparkium::MaterialShaderGraph, MaterialShaderGraph);
  return nullptr;
}

}  // namespace sparkium::render_shared
