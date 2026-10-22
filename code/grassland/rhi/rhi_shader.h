#pragma once
#include "grassland/rhi/rhi_object.h"

namespace grassland::rhi {
struct ShaderDesc {
  std::string source;
  std::string entry_point;
  ShaderStage stage{SHADER_STAGE_VERTEX};
};

// Source only. Backend compilation occurs when creating a Program.
class IShader : public IObject {
 public:
  virtual const ShaderDesc &GetDesc() const = 0;

 protected:
  ~IShader() override;
};
}  // namespace grassland::rhi
