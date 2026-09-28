#pragma once
#include "grassland/graphics/backend/vulkan/helper/vulkan_util.h"

namespace grassland::graphics::backend {
class VulkanShader;
}

namespace grassland::graphics::backend::vulkan {
struct HitGroup {
  VulkanShader *closest_hit_shader{nullptr};
  VulkanShader *any_hit_shader{nullptr};
  VulkanShader *intersection_shader{nullptr};
  bool procedure{false};
};
}  // namespace grassland::graphics::backend::vulkan
