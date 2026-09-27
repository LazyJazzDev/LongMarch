#pragma once
#include "sparkium/core/core_util.h"
#include "sparkium/core/data_update_tracker.h"

namespace sparkium {
class Core : public Object {
 public:
  Core(graphics::Core *core);

  DataUpdateTracker &GetDataUpdateTracker() {
    return data_updates_;
  }

  int CreateBuffer(size_t size, graphics::BufferType type, double_ptr<graphics::Buffer> buffer);
  int CreateImage(int width, int height, graphics::ImageFormat format, double_ptr<graphics::Image> image);

  graphics::Core *GraphicsCore() const;

  // Resolve automatic selection and supported fallbacks for rendering and UI display.
  RenderPipeline ResolveRenderPipeline(RenderPipeline render_pipeline) const;

  void Render(Scene *scene, Camera *camera, Film *film, RenderPipeline render_pipeline = RENDER_PIPELINE_AUTO);

  const VirtualFileSystem &GetShadersVFS() const;

  graphics::Shader *GetShader(const std::string &name);

  graphics::ComputeProgram *GetComputeProgram(const std::string &name);

  graphics::Buffer *GetBuffer(const std::string &name);

  graphics::Image *GetImage(const std::string &name);

  void SetPublicResource(const std::string &name, std::unique_ptr<graphics::Shader> &&shader);
  void SetPublicResource(const std::string &name, std::unique_ptr<graphics::ComputeProgram> &&program);
  void SetPublicResource(const std::string &name, std::unique_ptr<graphics::Buffer> &&buffer);
  void SetPublicResource(const std::string &name, std::unique_ptr<graphics::Image> &&image);

 private:
  void LoadPublicShaders();
  void LoadPublicBuffers();
  void LoadPublicImages();

  graphics::Core *core_{nullptr};
  DataUpdateTracker data_updates_;

  VirtualFileSystem shaders_vfs_;

  std::map<std::string, std::unique_ptr<graphics::Shader>> shaders_;
  std::map<std::string, std::unique_ptr<graphics::ComputeProgram>> compute_programs_;
  std::map<std::string, std::unique_ptr<graphics::Buffer>> buffers_;
  std::map<std::string, std::unique_ptr<graphics::Image>> images_;
};
}  // namespace sparkium
