#pragma once
#include "grassland/rhi/rhi_shader.h"
#include "grassland/rhi/rhix/rhix_object_container.h"

namespace grassland::rhix {
class Shader : public ObjectContainer {
 public:
  Shader() noexcept;
  Shader(rhi::IShader *object) noexcept;
  rhi::IShader *Get() const noexcept;
  const rhi::ShaderDesc &GetDesc() const;
};
}  // namespace grassland::rhix
