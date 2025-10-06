#pragma once
#include "practium/entity/entity_util.h"

namespace practium {

class EntityPBDRigid : public Entity {
 public:
  EntityPBDRigid(Scene *scene, Model *model, MaterialPBDRigid *material);

  ~EntityPBDRigid() override = default;

  void SyncRenderState() const override;

  void SetPosition(const Vector3<float> &x);
  void SetOrientation(const Quaternion<float> &q);
  void SetMass(float mass);
  void SetInertia(float inertia);
  void SetVelocity(const Vector3<float> &v);
  void SetAngularVelocity(const Vector3<float> &w);

 private:
  std::unique_ptr<sparkium::GeometryMesh> geometry_mesh_;
  sparkium::Material *material_;
  std::unique_ptr<sparkium::EntityGeometryMaterial> entity_geometry_material_;
  int pbd_rigid_id_{-1};
};

}  // namespace practium
