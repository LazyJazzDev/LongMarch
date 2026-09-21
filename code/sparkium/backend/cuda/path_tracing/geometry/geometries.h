#pragma once
#include "sparkium/backend/cuda/path_tracing/geometry/geometry_mesh.h"

namespace sparkium::cuda_tracing {

Geometry *DedicatedCast(sparkium::Geometry *geometry);

}
