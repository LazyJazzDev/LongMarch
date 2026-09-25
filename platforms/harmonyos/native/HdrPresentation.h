#pragma once

#include "grassland/graphics/backend/vulkan/vulkan_core.h"

namespace longmarch::harmony {
class HdrPresentation {
 public:
  explicit HdrPresentation(grassland::graphics::backend::VulkanCore *core);
  grassland::graphics::Image *Convert(grassland::graphics::Image *source, bool hdr10, bool encoded_particles);

 private:
  grassland::graphics::backend::VulkanCore *core_;
  std::unique_ptr<grassland::graphics::Shader> shader_;
  std::unique_ptr<grassland::graphics::ComputeProgram> program_;
  std::unique_ptr<grassland::graphics::Buffer> settings_;
  std::unique_ptr<grassland::graphics::Image> output_;
};
}  // namespace longmarch::harmony
