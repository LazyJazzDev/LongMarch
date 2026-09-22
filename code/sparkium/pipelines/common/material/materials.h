#pragma once
#include "sparkium/pipelines/common/material/material_lambertian.h"
#include "sparkium/pipelines/common/material/material_light.h"
#include "sparkium/pipelines/common/material/material_principled.h"
#include "sparkium/pipelines/common/material/material_shader_graph.h"
#include "sparkium/pipelines/common/material/material_specular.h"

namespace sparkium::render_shared {

Material *DedicatedCast(sparkium::Material *material);

}
