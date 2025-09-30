#pragma once
#include "sparkium/pipelines/raytracing/core/core_util.h"

namespace sparkium::raytracing {

class Light : public Object {
 public:
  Light(Core *core);
  virtual ~Light() = default;
  virtual int SamplerShader(Scene *scene) = 0;
  virtual graphics::Buffer *SamplerData() = 0;
  virtual uint32_t SamplerPreprocess(graphics::CommandContext *cmd_ctx) = 0;
  operator bool() const {
    return core_ != nullptr;
  }

 protected:
  Core *core_;
};

}  // namespace sparkium::raytracing
