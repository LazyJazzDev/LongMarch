#include "practium/core/scene.h"

#include "practium/core/core.h"
#include "practium/entity/entities.h"
#include "practium/material/material_pbd_rigid.h"

namespace practium {

Scene::Scene(Core *core) : core_(core) {
  scene_ = std::make_unique<sparkium::Scene>(core_->GetRenderCore());
}

Core *Scene::GetCore() const {
  return core_;
}

sparkium::Scene *Scene::GetRenderScene() const {
  return scene_.get();
}

contradium::PBDSolver *Scene::GetPBDSolver() {
  if (!pbd_solver_) {
    pbd_solver_ = std::make_unique<contradium::PBDSolver>();
  }
  return pbd_solver_.get();
}

std::unique_ptr<Entity> Scene::AddEntity(Model *model, Material *material) {
  if (auto mat = dynamic_cast<MaterialPBDRigid *>(material)) {
    return std::make_unique<EntityPBDRigid>(this, model, mat);
  }
  return nullptr;
}

void Scene::RegisterEntity(Entity *entity) {
  entities_.push_back(entity);
}

void Scene::Step() {
  if (pbd_solver_) {
    pbd_solver_->Step(1e-2f);
  }

  for (auto entity : entities_) {
    entity->SyncRenderState();
  }
}

}  // namespace practium
