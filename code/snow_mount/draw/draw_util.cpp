#include "snow_mount/draw/draw_util.h"

namespace snow_mount::draw {

Transform PixelCoordToNDC(int width, int height) {
  Transform transform(1.0f);
  transform[0][0] = 2.0f / width;
  transform[1][1] = -2.0f / height;
  transform[3][0] = -1.0f;
  transform[3][1] = 1.0f;
  return transform;
}

}  // namespace snow_mount::draw
