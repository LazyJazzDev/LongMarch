#pragma once
#include "sparks/core/core_util.h"

namespace sparks {
class Core {
 public:
  Core(graphics::Core *core);

  graphics::Core *GraphicsCore() const;

  VirtualFileSystem GetShadersVFS() const;

  void ConvertFilmToImage(const Film &film, graphics::Image *image);

 private:
  graphics::Core *core_{nullptr};

  VirtualFileSystem shaders_vfs_;

  std::unique_ptr<graphics::Shader> film2img_shader_;
  std::unique_ptr<graphics::ComputeProgram> film2img_program_;
};
}  // namespace sparks
