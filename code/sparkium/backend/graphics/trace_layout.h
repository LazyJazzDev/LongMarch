#pragma once
#include <cstdint>
#include <glm/glm.hpp>

namespace sparkium::backend::graphics_backend {
struct GPUInstance {
  glm::mat4x3 object_to_world;
  glm::mat4x3 world_to_object;
  uint32_t root, geometry, material, primitive_count;
};

static_assert(sizeof(GPUInstance) == 112, "HLSL software instance layout changed");
}  // namespace sparkium::backend::graphics_backend
