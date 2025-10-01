#include "sparkium/core/scene.h"

#include "sparkium/pipelines/raster/core/core.h"
#include "sparkium/pipelines/raster/core/scene.h"

namespace sparkium::raster {

Scene::Scene(sparkium::Scene &scene) : scene_(scene), core_(DedicatedCast(scene.GetCore())) {
}

void Scene::Render(Camera *camera, Film *film) {
  LogInfo("Rendering with camera: {} film: {}", (void *)camera, (void *)film);
}

Scene *DedicatedCast(sparkium::Scene *scene) {
  COMPONENT_CAST(scene, Scene);
}

}  // namespace sparkium::raster
