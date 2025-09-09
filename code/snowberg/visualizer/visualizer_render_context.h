#pragma once
#include "snowberg/visualizer/visualizer_util.h"

namespace snowberg::visualizer {
struct RenderContext {
  graphics::CommandContext *cmd_ctx;
  Film *film;
  graphics::Buffer *camera_buffer;
  OwnershipHolder *ownership_holder;
};
}  // namespace snowberg::visualizer
