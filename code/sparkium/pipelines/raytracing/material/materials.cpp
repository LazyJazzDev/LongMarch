#include "sparkium/pipelines/raytracing/material/materials.h"

namespace sparkium::raytracing {

Material *DedicatedCast(sparkium::Material *material) {
  DEDICATED_CAST(material, sparkium::MaterialLambertian, MaterialLambertian);
  DEDICATED_CAST(material, sparkium::MaterialLight, MaterialLight);
  DEDICATED_CAST(material, sparkium::MaterialSpecular, MaterialSpecular);
  DEDICATED_CAST(material, sparkium::MaterialPrincipled, MaterialPrincipled);
  return nullptr;
}

}  // namespace sparkium::raytracing
