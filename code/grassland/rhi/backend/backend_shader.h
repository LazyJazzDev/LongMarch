#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_shader.h"

namespace grassland::rhi::backend {
class Shader final : public Object<IShader> {
 public:
  explicit Shader(const ShaderDesc &desc);
  const ShaderDesc &GetDesc() const override;

 private:
  ShaderDesc desc_;
};
}  // namespace grassland::rhi::backend
