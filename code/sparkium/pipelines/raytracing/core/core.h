#pragma once
#include "sparkium/core/core_util.h"

namespace sparkium::raytracing {
void Render(sparkium::Core *core,
            sparkium::Scene *scene,
            sparkium::Camera *camera,
            sparkium::Film *film,
            bool software = false,
            bool ray_query = false);
}  // namespace sparkium::raytracing
