#pragma once
#include "sparkium/pipelines/realtime/core/core_util.h"

namespace sparkium::realtime {

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
  VirtualFileSystem shaders_vfs_;
  std::map<std::string, std::unique_ptr<graphics::Shader>> shaders_;
  std::map<std::string, std::unique_ptr<graphics::ComputeProgram>> compute_programs_;
};

Core *DedicatedCast(sparkium::Core *core);

}  // namespace sparkium::realtime
