#include "sparkium/pipelines/realtime/realtime.h"

#include "sparkium/pipelines/common/core/core.h"
#include "sparkium/pipelines/realtime/scene.h"

namespace sparkium::realtime {
namespace {
Scene *DedicatedCast(sparkium::Scene *scene) {
  COMPONENT_CAST(scene, Scene);
}
}  // namespace

void Render(sparkium::Core *core, sparkium::Scene *scene, sparkium::Camera *camera, sparkium::Film *film) {
  render_shared::DedicatedCast(core);
  DedicatedCast(scene)->Render(camera, film);
}
}  // namespace sparkium::realtime
