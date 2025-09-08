#pragma once
#include "snow_mount/visualizer/visualizer_util.h"

namespace XS::visualizer {
struct RenderContext {
  graphics::CommandContext *cmd_ctx;
  Film *film;
  graphics::Buffer *camera_buffer;
  OwnershipHolder *ownership_holder;
};
}  // namespace XS::visualizer
