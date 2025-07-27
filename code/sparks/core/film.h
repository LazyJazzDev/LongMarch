#pragma once
#include "core.h"
#include "sparks/core/core_util.h"

namespace sparks {

class Film {
 public:
  Film(Core *core, int width, int height);

  void Reset();

  int GetWidth() const;
  int GetHeight() const;

 private:
  friend Scene;
  friend Core;
  std::unique_ptr<graphics::Image> accumulated_color_;
  std::unique_ptr<graphics::Image> accumulated_samples_;
  Core *core_;
};

}  // namespace sparks
