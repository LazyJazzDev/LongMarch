#pragma once
#include "sparkium/pipelines/raytracing/core/core_util.h"

namespace sparkium::raytracing {

class Core : public Object {
 public:
  Core(sparkium::Core &core);

  graphics::Core *GraphicsCore() const;

  const VirtualFileSystem &GetShadersVFS() const;

  graphics::Shader *GetShader(const std::string &name);

  graphics::ComputeProgram *GetComputeProgram(const std::string &name);

  graphics::Image *GetImage(const std::string &name);

  graphics::Buffer *GetBuffer(const std::string &name);

 private:
  void LoadPublicShaders();

  sparkium::Core &core_;
};

Core *DedicatedCast(sparkium::Core *core);

void Render(sparkium::Core *core, sparkium::Scene *scene, sparkium::Camera *camera, sparkium::Film *film);
}  // namespace sparkium::raytracing
