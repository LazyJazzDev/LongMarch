#pragma once
#include "sparkium/backend/cuda/path_tracing/entity/entity_geometry_meterial.h"
#include "sparkium/backend/cuda/path_tracing/entity/entity_point_light.h"

namespace sparkium::cuda_tracing {
Entity *DedicatedCast(sparkium::Entity *entity);
}
