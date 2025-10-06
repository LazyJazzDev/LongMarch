#include "practium/entity/entity_pbd_rigid.h"

#include "practium/core/core.h"
#include "practium/core/scene.h"
#include "practium/material/material_pbd_rigid.h"

namespace practium {

EntityPBDRigid::EntityPBDRigid(Scene *scene, Model *model, MaterialPBDRigid *material) : Entity(scene) {
  geometry_mesh_ = std::make_unique<sparkium::GeometryMesh>(scene_->GetCore()->GetRenderCore(), model->VisualMesh());
  material_ = model->VisualMaterial();
  entity_geometry_material_ = std::make_unique<sparkium::EntityGeometryMaterial>(scene_->GetCore()->GetRenderCore(),
                                                                                 geometry_mesh_.get(), material_);
  scene_->GetRenderScene()->AddEntity(entity_geometry_material_.get());
  pbd_rigid_id_ = scene_->GetPBDSolver()->AddEntity(model->CollisionMesh(), {0.0f, 0.0f, 0.0f},
                                                    {1.0f, 0.0f, 0.0f, 0.0f}, material->mass, material->inertia);
}

void EntityPBDRigid::SyncRenderState() const {
  auto &pbd_entity = scene_->GetPBDSolver()->GetEntity(pbd_rigid_id_);
  Matrix<float, 3, 4> transform;
  transform.block(0, 0, 3, 3) = pbd_entity.q_.toRotationMatrix();
  transform.col(3) = pbd_entity.x_;
  entity_geometry_material_->transform = EigenToGLM(transform);
}

void EntityPBDRigid::SetPosition(const Vector3<float> &x) {
  scene_->GetPBDSolver()->SetPosition(pbd_rigid_id_, x);
}

void EntityPBDRigid::SetOrientation(const Quaternion<float> &q) {
  scene_->GetPBDSolver()->SetOrientation(pbd_rigid_id_, q);
}

void EntityPBDRigid::SetMass(float mass) {
  scene_->GetPBDSolver()->SetMass(pbd_rigid_id_, mass);
}

void EntityPBDRigid::SetInertia(float inertia) {
  scene_->GetPBDSolver()->SetInertia(pbd_rigid_id_, inertia);
}

void EntityPBDRigid::SetVelocity(const Vector3<float> &v) {
  scene_->GetPBDSolver()->SetVelocity(pbd_rigid_id_, v);
}

void EntityPBDRigid::SetAngularVelocity(const Vector3<float> &w) {
  scene_->GetPBDSolver()->SetAngularVelocity(pbd_rigid_id_, w);
}

}  // namespace practium
