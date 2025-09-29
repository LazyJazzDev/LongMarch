#pragma once
#include "sparkium/core/core_util.h"

namespace sparkium {

class Scene : public Object {
 public:
  Scene(Core *core);

  Core *GetCore() const;

  void AddEntity(Entity *entity);

  void DeleteEntity(Entity *entity);

  void SetEntityActive(Entity *entity, bool active);

  struct Settings {
    int samples_per_dispatch = 128;
    int max_bounces = 32;
    int alpha_shadow = false;
  } settings;

  struct EntityStatus {
    bool active{true};
  };

  const std::map<Entity *, EntityStatus> &GetEntities() const;

 private:
  Core *core_;
  std::map<Entity *, EntityStatus> entities_;
};

}  // namespace sparkium
