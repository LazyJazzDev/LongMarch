#pragma once
#include "sparkium/backend/cuda/path_tracing/core/core_util.h"

namespace sparkium::cuda_tracing {

class Core : public Object {
 public:
  Core(sparkium::Core &core);

  backend::Device *BackendDevice() const;

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

void Render(sparkium::Core *core,
            sparkium::Scene *scene,
            sparkium::Camera *camera,
            sparkium::Film *film,
            bool optix = false);
}  // namespace sparkium::cuda_tracing
