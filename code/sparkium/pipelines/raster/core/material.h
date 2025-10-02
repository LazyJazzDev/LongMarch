#pragma once
#include "sparkium/pipelines/raster/core/core_util.h"

namespace sparkium::raster {

class Material : public Object {
 public:
  explicit Material(Core *core) : core_(core) {
  }
  virtual ~Material() = default;
  virtual graphics::Shader *PixelShader() = 0;
  virtual graphics::Buffer *Buffer() = 0;
  virtual void Sync() = 0;
  virtual glm::vec3 Emission() const;

 protected:
  Core *core_{};
};

}  // namespace sparkium::raster
