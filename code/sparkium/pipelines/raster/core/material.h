#pragma once
#include "sparkium/pipelines/raster/core/core_util.h"

namespace sparkium::raster {

class Material : public Object {
 public:
  explicit Material(Core *core) : core_(core) {
  }

  virtual ~Material() = default;
  virtual rhi::IShader *PixelShader() = 0;
  virtual void SetupProgram(rhi::ProgramDesc &program);
  virtual void BindMaterialResources(rhi::IBindingGroup *group);
  virtual void Sync() = 0;
  virtual glm::vec3 Emission() const;

 protected:
  Core *core_{};
};

}  // namespace sparkium::raster
