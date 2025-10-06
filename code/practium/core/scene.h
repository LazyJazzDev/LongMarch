#pragma once
#include "practium/core/core_util.h"

namespace practium {

class Scene {
 public:
  struct Settings {
    float dt{0.003f};
  };

  Scene(Core *core);

  Core *GetCore() const;

  sparkium::Scene *GetRenderScene() const;

  contradium::PBDSolver *GetPBDSolver();

  std::unique_ptr<Entity> AddEntity(Model *model, Material *material);

  void RegisterEntity(Entity *entity);

  void Step();

  void SyncRenderState();

 private:
  Core *core_;
  std::unique_ptr<sparkium::Scene> scene_;
  std::unique_ptr<contradium::PBDSolver> pbd_solver_;

  std::vector<Entity *> entities_;
};

}  // namespace practium
