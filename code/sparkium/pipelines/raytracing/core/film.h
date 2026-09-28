#pragma once
#include "sparkium/pipelines/raytracing/core/core_util.h"

namespace sparkium::raytracing {

class Film : public Object {
 public:
  Film(sparkium::Film &film);
  void Reset();

  int GetWidth() const;
  int GetHeight() const;

 private:
  sparkium::Film &film_;
  Core *core_;

  friend Scene;
  friend Core;
  std::unique_ptr<graphics::Image> accumulated_color_;
  std::unique_ptr<graphics::Image> accumulated_samples_;
};

Film *DedicatedCast(sparkium::Film *film);

}  // namespace sparkium::raytracing
