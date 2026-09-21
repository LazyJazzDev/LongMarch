#pragma once
#include "sparkium/backend/common/path_tracing/entity/entity_geometry_meterial.h"
#include "sparkium/backend/common/path_tracing/entity/entity_point_light.h"

namespace sparkium::raytracing {
Entity *DedicatedCast(sparkium::Entity *entity);
}
