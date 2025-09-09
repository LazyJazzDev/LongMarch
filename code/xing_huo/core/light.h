#pragma once
#include "xing_huo/core/core_util.h"

namespace XH {
class Light {
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
}  // namespace XH
