#pragma once
#include "sparkium/pipelines/common/core/core_util.h"

namespace sparkium::render_shared {

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

}  // namespace sparkium::render_shared
