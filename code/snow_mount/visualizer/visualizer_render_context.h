#pragma once
#include "snow_mount/visualizer/visualizer_util.h"

namespace snow_mount::visualizer {
struct RenderContext {
  graphics::CommandContext *cmd_ctx;
  Film *film;
  graphics::Buffer *camera_buffer;
};
}  // namespace snow_mount::visualizer
