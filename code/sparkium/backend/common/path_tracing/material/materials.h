#pragma once
#include "sparkium/backend/common/path_tracing/material/material_lambertian.h"
#include "sparkium/backend/common/path_tracing/material/material_light.h"
#include "sparkium/backend/common/path_tracing/material/material_principled.h"
#include "sparkium/backend/common/path_tracing/material/material_shader_graph.h"
#include "sparkium/backend/common/path_tracing/material/material_specular.h"

namespace sparkium::raytracing {

Material *DedicatedCast(sparkium::Material *material);

}
