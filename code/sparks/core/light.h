#pragma once
#include "sparks/core/core_util.h"

namespace sparks {
class Light {
 public:
  Light(Core *core);
  virtual ~Light() = default;
  virtual graphics::Buffer *SamplerData() = 0;
  virtual uint32_t SamplerPreprocess(graphics::CommandContext *cmd_ctx) = 0;
  virtual const CodeLines &SamplerImpl() const = 0;
  operator bool() const {
    return core_ != nullptr;
  }

 protected:
  Core *core_;
};
}  // namespace sparks
