#pragma once
#include "core.h"
#include "sparks/core/core_util.h"

namespace XH {

class Film {
 public:
  Film(Core *core, int width, int height);

  void Reset();

  int GetWidth() const;
  int GetHeight() const;

  struct Info {
    int accumulated_samples{0};
    float persistence{1.0};
    float clamping{100.0f};
    float max_exposure{1.0f};
  } info;

 private:
  friend Scene;
  friend Core;
  std::unique_ptr<graphics::Image> accumulated_color_;
  std::unique_ptr<graphics::Image> accumulated_samples_;
  Core *core_;
};

}  // namespace XH
