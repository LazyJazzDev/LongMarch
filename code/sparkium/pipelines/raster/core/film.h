#pragma once
#include "sparkium/pipelines/raster/core/core_util.h"

namespace sparkium::raster {

class Film : public Object {
 public:
  Film(sparkium::Film &film);

  rhi::IImage *GetAlbedoRoughnessBuffer() const;
  rhi::IImage *GetPositionSpecularBuffer() const;
  rhi::IImage *GetNormalMetallicBuffer() const;

 private:
  friend Scene;
  friend Core;
  sparkium::Film &film_;
  Core *core_;

  rhi::ObjPtr<rhi::IImage> albedo_roughness_buffer_;
  rhi::ObjPtr<rhi::IImage> position_specular_buffer_;
  rhi::ObjPtr<rhi::IImage> normal_metallic_buffer_;
};

Film *DedicatedCast(sparkium::Film *film);

}  // namespace sparkium::raster
