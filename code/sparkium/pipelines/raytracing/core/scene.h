#pragma once
#include "sparkium/pipelines/common/core/scene.h"

namespace sparkium::raytracing {
class Film;

class Scene : public render_shared::Scene {
 public:
  explicit Scene(sparkium::Scene &scene) : render_shared::Scene(scene) {
  }

  void Render(render_shared::Camera *camera, Film *film, bool software = false, bool ray_query = false);

 private:
  bool rendered_{};
};

Scene *DedicatedCast(sparkium::Scene *scene);
}  // namespace sparkium::raytracing
