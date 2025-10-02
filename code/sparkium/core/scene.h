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
    struct RayTracing {
      int samples_per_dispatch = 128;
      int max_bounces = 32;
      int alpha_shadow = false;
    } raytracing;
    struct Rasterization {
      glm::vec3 ambient_light{0.1f, 0.1f, 0.1f};
    } raster;
    int &samples_per_dispatch{raytracing.samples_per_dispatch};
    int &max_bounces{raytracing.max_bounces};
    int &alpha_shadow{raytracing.alpha_shadow};
    glm::vec3 &ambient_light{raster.ambient_light};
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
