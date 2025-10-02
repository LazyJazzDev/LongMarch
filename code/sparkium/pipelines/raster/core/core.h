#pragma once
#include "sparkium/pipelines/raster/core/core_util.h"

namespace sparkium::raster {

class Core : public Object {
 public:
  Core(sparkium::Core &core);

  graphics::Core *GraphicsCore() const;

  const VirtualFileSystem &GetShadersVFS() const;

  graphics::Shader *GetShader(const std::string &name);

  graphics::ComputeProgram *GetComputeProgram(const std::string &name);

  graphics::Buffer *GetBuffer(const std::string &name);

  graphics::Image *GetImage(const std::string &name);

 private:
  sparkium::Core &core_;
};

void Render(sparkium::Core *core, sparkium::Scene *scene, sparkium::Camera *camera, sparkium::Film *film);

Core *DedicatedCast(sparkium::Core *core);

}  // namespace sparkium::raster
