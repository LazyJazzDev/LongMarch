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
  int32_t shader_index{-1};
  int32_t data_index{-1};
};

struct LightMetadata {
  int sampler_shader_index{-1};
  int sampler_data_index{-1};
  int custom_index{-1};
  uint32_t power_offset{0};
};

struct InstanceMetadata {
  int geometry_data_index{-1};
  int surface_shader_index{-1};
  int surface_data_index{-1};
  int custom_index{-1};
};

struct BlellochScanMetadata {
  uint32_t offset;
  uint32_t stride;
  uint32_t element_count;
  uint32_t padding[61];
};

}  // namespace sparks
