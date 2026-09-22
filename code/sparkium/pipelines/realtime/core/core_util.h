#pragma once
#include "sparkium/core/camera.h"
#include "sparkium/core/core.h"
#include "sparkium/core/film.h"
#include "sparkium/core/scene.h"
#include "sparkium/entity/entities.h"
#include "sparkium/geometry/geometries.h"
#include "sparkium/material/materials.h"

namespace sparkium::realtime {

class Core;
class Scene;
class Geometry;
class Material;
class Light;
class Entity;
class Camera;

struct LightMetadata {
  int sampler_shader_index{-1};
  int sampler_data_index{-1};
  int custom_index{-1};
  uint32_t power_offset{0};
};

struct InstanceMetadata {
  int geometry_data_index{-1};
  int material_data_index{-1};
  int custom_index{-1};
};

struct BlellochScanMetadata {
  uint32_t offset;
  uint32_t stride;
  uint32_t element_count;
  uint32_t padding[61];
};

}  // namespace sparkium::realtime
