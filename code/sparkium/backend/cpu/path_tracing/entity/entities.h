#pragma once
#include "sparkium/backend/cpu/path_tracing/entity/entity_geometry_meterial.h"
#include "sparkium/backend/cpu/path_tracing/entity/entity_point_light.h"

namespace sparkium::cpu_tracing {
Entity *DedicatedCast(sparkium::Entity *entity);
}
