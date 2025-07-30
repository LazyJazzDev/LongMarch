#pragma once
#include "grassland/grassland.h"
#include "sparks/core/code_lines.h"

namespace sparks {

using namespace grassland;

class Core;
class Geometry;
class Surface;
class Entity;
class Film;
class Scene;
class Camera;
class Light;

struct GeometryRegistration {
  int32_t hit_group_index;
  int32_t data_index;
  graphics::AccelerationStructure *blas;
};

struct InstanceRegistration {
  int32_t instance_index;
};

struct SurfaceRegistration {
  int32_t shader_index;
  int32_t buffer_index;
};

}  // namespace sparks
