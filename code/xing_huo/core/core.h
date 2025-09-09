#pragma once
#include "xing_huo/core/core_util.h"

namespace XH {
class Core {
 public:
  Core(graphics::Core *core);

  graphics::Core *GraphicsCore() const;

  const VirtualFileSystem &GetShadersVFS() const;

  void ConvertFilmToRawImage(const Film &film, graphics::Image *image);

  void ToneMapping(graphics::Image *raw_image, graphics::Image *output_image);

  graphics::Shader *GetShader(const std::string &name);

  graphics::ComputeProgram *GetComputeProgram(const std::string &name);

  graphics::Buffer *GetBuffer(const std::string &name);

  graphics::Image *GetImage(const std::string &name);

 private:
  void LoadPublicShaders();
  void LoadPublicBuffers();
  void LoadPublicImages();

  graphics::Core *core_{nullptr};

  VirtualFileSystem shaders_vfs_;

  std::map<std::string, std::unique_ptr<graphics::Shader>> shaders_;
  std::map<std::string, std::unique_ptr<graphics::ComputeProgram>> compute_programs_;
  std::map<std::string, std::unique_ptr<graphics::Buffer>> buffers_;
  std::map<std::string, std::unique_ptr<graphics::Image>> images_;
};
}  // namespace XH
