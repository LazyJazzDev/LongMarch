#pragma once
#include "sparkium/pipelines/raster/core/core_util.h"

namespace sparkium::raster {
void Render(sparkium::Core *core, sparkium::Scene *scene, sparkium::Camera *camera, sparkium::Film *film);
}
