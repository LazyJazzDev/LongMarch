#pragma once
#include "sparkium/pipelines/raster/core/core_util.h"

namespace sparkium::raster {

class Film : public Object {
 public:
  Film(sparkium::Film &film);

 private:
  sparkium::Film &film_;
  Core *core_;

  std::unique_ptr<graphics::Image> albedo_roughness_buffer_;
  std::unique_ptr<graphics::Image> position_specular_buffer_;
  std::unique_ptr<graphics::Image> normal_metallic_buffer_;
};

Film *DedicatedCast(sparkium::Film *film);

}  // namespace sparkium::raster
