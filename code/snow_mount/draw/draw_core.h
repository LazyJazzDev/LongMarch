#pragma once
#include "snow_mount/draw/draw_util.h"

namespace snow_mount::draw {

class Core {
 public:
  Core(graphics::Core *core);

  void BeginDraw();
  void EndDraw();
  void Render(graphics::Image *image);

  void CreateTexture()

      private : graphics::Core *core_;
};

}  // namespace snow_mount::draw
