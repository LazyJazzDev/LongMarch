#pragma once
#include "sparks/core/core_util.h"

namespace sparks {
class Core {
 public:
  Core(graphics::Core *core);

  graphics::Core *GraphicsCore() const;

  const VirtualFileSystem &GetShadersVFS() const;

  void ConvertFilmToRawImage(const Film &film, graphics::Image *image);

  void ToneMapping(graphics::Image *raw_image, graphics::Image *output_image);

  graphics::Shader *GetShader(const std::string &name);

  graphics::ComputeProgram *GetComputeProgram(const std::string &name);

  graphics::Buffer *SobolBuffer();

 private:
  void LoadPublicShaders();
  void LoadSobolBuffer();

  graphics::Core *core_{nullptr};

  VirtualFileSystem shaders_vfs_;

  std::map<std::string, std::unique_ptr<graphics::Shader>> shaders_;
  std::map<std::string, std::unique_ptr<graphics::ComputeProgram>> compute_programs_;
  std::unique_ptr<graphics::Buffer> sobol_buffer_;
};
}  // namespace sparks
