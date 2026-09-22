#include "sparkium/pipelines/realtime/realtime.h"

#include "sparkium/pipelines/realtime/core/core.h"
#include "sparkium/pipelines/realtime/core/scene.h"

namespace sparkium::realtime {
namespace {
Scene *DedicatedCast(sparkium::Scene *scene) {
  COMPONENT_CAST(scene, Scene);
}
}  // namespace

void Render(sparkium::Core *core, sparkium::Scene *scene, sparkium::Camera *camera, sparkium::Film *film) {
  DedicatedCast(core);
  DedicatedCast(scene)->Render(camera, film);
}
}  // namespace sparkium::realtime
