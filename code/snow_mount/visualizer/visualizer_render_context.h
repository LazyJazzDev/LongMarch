#pragma once
#include "snow_mount/visualizer/visualizer_util.h"

namespace snow_mount::visualizer {
struct RenderContext {
  graphics::CommandContext *cmd_ctx;
  Film *film;
};
}  // namespace snow_mount::visualizer
