#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_shader.h"

namespace grassland::rhi::backend {
class Shader final : public Object<IShader> {
 public:
  static Shader *Create(const ShaderDesc &desc, BackendAPI api);
  const ShaderDesc &CompileDesc() const;
  const ShaderDesc &GetDesc() const override;

 private:
  explicit Shader(const ShaderDesc &desc);
  ShaderDesc desc_;
  ShaderDesc compiled_desc_;
};
}  // namespace grassland::rhi::backend
