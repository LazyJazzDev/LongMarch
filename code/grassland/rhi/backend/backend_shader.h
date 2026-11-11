#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_shader.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::IBackendShader)

namespace grassland::rhi::backend {
class IBackendShader : public IShader {
 public:
  static IBackendShader *Create(const ShaderDesc &desc, BackendAPI api);
  virtual ShaderStage GetStage() const = 0;
  virtual const std::string &GetEntry() const = 0;
  virtual const std::string &GetPreprocessedCode() const = 0;
};

;
}  // namespace grassland::rhi::backend
