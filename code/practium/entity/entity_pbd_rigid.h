#pragma once
#include "practium/entity/entity_util.h"

namespace practium {

class EntityPBDRigid : public Entity {
 public:
  EntityPBDRigid(Scene *scene, Model *model, MaterialPBDRigid *material);

  virtual ~EntityPBDRigid() = default;

 private:
  std::unique_ptr<sparkium::GeometryMesh> geometry_mesh_;
  sparkium::Material *material_;
  std::unique_ptr<sparkium::EntityGeometryMaterial> entity_geometry_material_;
  int pbd_rigid_id_{-1};
};

}  // namespace practium
