#pragma once
#include "sparkium/pipelines/raytracing/material/material_lambertian.h"
#include "sparkium/pipelines/raytracing/material/material_principled.h"
#include "sparkium/pipelines/raytracing/material/material_specular.h"

namespace sparkium::raytracing {

Material *DedicatedCast(sparkium::Material *material);

}
