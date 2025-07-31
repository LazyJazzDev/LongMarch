#pragma once
#include "sparks/core/core_util.h"

namespace sparks {
class Core {
 public:
  Core(graphics::Core *core);

  graphics::Core *GraphicsCore() const;

  const VirtualFileSystem &GetShadersVFS() const;

  void ConvertFilmToImage(const Film &film, graphics::Image *image);

  graphics::Shader *GetShader(const std::string &name);

  graphics::ComputeProgram *GetComputeProgram(const std::string &name);

 private:
  graphics::Core *core_{nullptr};

  VirtualFileSystem shaders_vfs_;

  std::unique_ptr<graphics::Shader> film2img_shader_;
  std::unique_ptr<graphics::ComputeProgram> film2img_program_;
  std::map<std::string, std::unique_ptr<graphics::Shader>> shaders_;
  std::map<std::string, std::unique_ptr<graphics::ComputeProgram>> compute_programs_;
};
}  // namespace sparks
