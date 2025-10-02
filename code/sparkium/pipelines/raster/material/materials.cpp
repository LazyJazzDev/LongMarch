#include "sparkium/pipelines/raster/material/materials.h"

namespace sparkium::raster {

Material *DedicatedCast(sparkium::Material *material) {
  DEDICATED_CAST(material, sparkium::MaterialLambertian, MaterialLambertian)
  DEDICATED_CAST(material, sparkium::MaterialLight, MaterialLight)
  DEDICATED_CAST(material, sparkium::MaterialPrincipled, MaterialPrincipled)
  DEDICATED_CAST(material, sparkium::MaterialSpecular, MaterialSpecular)
  return nullptr;
}

}  // namespace sparkium::raster
