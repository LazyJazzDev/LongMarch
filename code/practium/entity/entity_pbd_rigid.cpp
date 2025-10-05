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
  scene_->GetPBDSolver()->AddEntity(model->CollisionMesh(), {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f},
                                    material->mass, material->inertia);
}

}  // namespace practium
