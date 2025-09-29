#pragma once
#include "sparkium/core/core_util.h"

namespace sparkium {

class Scene : public Object {
 public:
  Scene(Core *core);

  void Render(Camera *camera, Film *film);

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
    int shader_version{0};
  };

 private:
  Core *core_;
  std::map<Entity *, EntityStatus> entities_;
};

}  // namespace sparkium
