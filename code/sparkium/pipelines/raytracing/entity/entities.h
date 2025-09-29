#pragma once
#include "sparkium/pipelines/raytracing/entity/entity_area_light.h"
#include "sparkium/pipelines/raytracing/entity/entity_geometry_light.h"
#include "sparkium/pipelines/raytracing/entity/entity_geometry_meterial.h"
#include "sparkium/pipelines/raytracing/entity/entity_point_light.h"

namespace sparkium::raytracing {
Entity *DedicatedCast(sparkium::Entity *entity);
}
