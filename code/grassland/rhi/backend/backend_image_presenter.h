#pragma once
#include <map>
#include <tuple>

#include "grassland/rhi/rhi.h"

namespace grassland::rhi::backend {
class Device;

// Records a full-image color conversion. Cached pipelines do not retain source images.
class ImagePresenter {
 public:
  explicit ImagePresenter(Device *device);
  bool Record(ICommandContext *commands,
              IImage *source,
              IImage *target,
              ColorSpace source_space,
              ColorSpace target_space,
              float white_scale);

 private:
  Device *device_;
  std::map<std::tuple<ImageFormat, ColorSpace, ColorSpace>, ObjPtr<IProgram>> programs_;
  ObjPtr<IBuffer> settings_;
  ObjPtr<ISampler> sampler_;
};
}  // namespace grassland::rhi::backend
