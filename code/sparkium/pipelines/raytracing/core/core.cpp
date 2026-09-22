#include "sparkium/pipelines/raytracing/core/core.h"

#include "sparkium/pipelines/common/core/camera.h"
#include "sparkium/pipelines/common/core/core.h"
#include "sparkium/pipelines/raytracing/core/film.h"
#include "sparkium/pipelines/raytracing/core/scene.h"

namespace sparkium::raytracing {
void Render(sparkium::Core *core,
            sparkium::Scene *scene,
            sparkium::Camera *camera,
            sparkium::Film *film,
            bool software,
            bool ray_query) {
  render_shared::DedicatedCast(core);
  DedicatedCast(scene)->Render(render_shared::DedicatedCast(camera), DedicatedCast(film), software, ray_query);
}
}  // namespace sparkium::raytracing
