#pragma once
#include "sparkium/pipelines/realtime/material/material_lambertian.h"
#include "sparkium/pipelines/realtime/material/material_light.h"
#include "sparkium/pipelines/realtime/material/material_principled.h"
#include "sparkium/pipelines/realtime/material/material_shader_graph.h"
#include "sparkium/pipelines/realtime/material/material_specular.h"

namespace sparkium::realtime {

Material *DedicatedCast(sparkium::Material *material);

}
