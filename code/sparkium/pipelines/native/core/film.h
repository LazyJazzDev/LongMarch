#pragma once

// Host-side accumulation buffers for the native backends.
//
// `raytracing::Film` keeps `accumulated_color_` and `accumulated_samples_` as
// GPU images. The native backends keep the very same two planes in host memory
// and hook the same `Film::RegisterResetCallback`, so film persistence,
// sample counting and resets behave identically.

#include <vector>

#include "sparkium/core/film.h"
#include "sparkium/pipelines/native/shared/native_integrator.h"

namespace sparkium::native {

class Film : public Object {
 public:
  explicit Film(sparkium::Film &film);

  void Reset();

  int GetWidth() const;
  int GetHeight() const;

  std::vector<float4> &AccumulatedColor() {
    return accumulated_color_;
  }
  std::vector<float> &AccumulatedSamples() {
    return accumulated_samples_;
  }

  sparkium::Film &Owner() {
    return film_;
  }

  // Uploads `film2img.hlsl`'s result into the film's raw image, so
  // `Film::Develop`, the GUI and the existing tests read a native render the
  // same way they read a GPU one.
  void PublishRawImage();

 private:
  sparkium::Film &film_;
  std::vector<float4> accumulated_color_;
  std::vector<float> accumulated_samples_;
};

Film *DedicatedCast(sparkium::Film *film);

}  // namespace sparkium::native
