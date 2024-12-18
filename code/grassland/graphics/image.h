#pragma once
#include "grassland/graphics/graphics_util.h"

namespace grassland::graphics {

class Image {
 public:
  virtual ~Image() = default;
  virtual Extent2D Extent() const = 0;
  virtual ImageFormat Format() const = 0;
};

}  // namespace grassland::graphics
