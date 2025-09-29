#pragma once
#include "core.h"
#include "sparkium/core/core_util.h"

namespace sparkium {

class Film : public Object {
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

  void Develop(graphics::Image *targ_image);

 private:
  Core *core_;
  graphics::Extent2D extent_;
};

}  // namespace sparkium
