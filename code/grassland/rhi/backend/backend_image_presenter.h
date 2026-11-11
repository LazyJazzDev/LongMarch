#pragma once
#include <map>
#include <tuple>

#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi.h"

namespace grassland::rhi::backend {
// Records a full-image color conversion. Cached pipelines do not retain source images.
class ImagePresenter final {
 public:
  explicit ImagePresenter(IBackendDevice *device);
  bool Record(ICommandContext *commands,
              IImage *source,
              IImage *target,
              ColorSpace source_space,
              ColorSpace target_space,
              float white_scale);

 private:
  IBackendDevice *device_;
  std::map<std::tuple<ImageFormat, ColorSpace, ColorSpace>, ObjPtr<IProgram>> programs_;
  ObjPtr<IBuffer> settings_;
  ObjPtr<ISampler> sampler_;
};
}  // namespace grassland::rhi::backend
