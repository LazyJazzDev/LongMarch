#include "sparkium/pipelines/realtime/entity/entity_geometry_meterial.h"

#include "sparkium/pipelines/realtime/core/core.h"
#include "sparkium/pipelines/realtime/core/scene.h"
#include "sparkium/pipelines/realtime/geometry/geometries.h"
#include "sparkium/pipelines/realtime/material/materials.h"

namespace sparkium::realtime {

EntityGeometryMaterial::EntityGeometryMaterial(sparkium::EntityGeometryMaterial &entity)
    : entity_(entity),
      Entity(DedicatedCast(entity.GetCore())) {
  geometry_ = DedicatedCast(entity_.GetGeometry());
  material_ = DedicatedCast(entity_.GetMaterial());

  if (!geometry_ || !material_)
    throw std::runtime_error("realtime GI requires supported geometry and material components");

  light_geom_mat_ = std::make_unique<LightGeometryMaterial>(core_, geometry_, material_, entity_.transform);
}

void EntityGeometryMaterial::Update(Scene *scene) {
  if (!geometry_ || !material_ || geometry_->PrimitiveCount() == 0)
    return;
  material_->Update(scene);
  int32_t light_index = scene->RegisterLight(light_geom_mat_.get());
  if (!dynamic_cast<GeometryMesh *>(geometry_))
    throw std::runtime_error("realtime GI currently requires triangle geometry");
  const int32_t instance_index =
      scene->RegisterSoftwareInstance(geometry_, material_, entity_.GetTransformation(), light_index);
  scene->LightCustomIndex(light_index) = instance_index;
}

}  // namespace sparkium::realtime
