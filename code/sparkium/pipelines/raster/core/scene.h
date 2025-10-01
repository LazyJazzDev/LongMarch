#pragma once
#include "sparkium/pipelines/raster/core/core_util.h"

namespace sparkium::raster {

class Scene : public Object {
 public:
  Scene(sparkium::Scene &scene);

  void Render(Camera *camera, Film *film);

 private:
  sparkium::Scene &scene_;
  Core *core_;
};

Scene *DedicatedCast(sparkium::Scene *scene);

}  // namespace sparkium::raster
