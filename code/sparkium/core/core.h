#pragma once
#include "sparkium/core/core_util.h"

namespace sparkium {
class Core : public Object {
 public:
  // Borrowed device; destroy scenes, resources and this Core before the device.
  explicit Core(rhi::IDevice *device);

  rhi::IDevice *GetDevice() const;

  // Resolve automatic selection and supported fallbacks for rendering and UI display.
  RenderPipeline ResolveRenderPipeline(RenderPipeline render_pipeline) const;

  void Render(Scene *scene, Camera *camera, Film *film, RenderPipeline render_pipeline = RENDER_PIPELINE_AUTO);

  const VirtualFileSystem &GetShadersVFS() const;

  rhi::IShader *GetShader(const std::string &name);

  rhi::IComputeProgram *GetComputeProgram(const std::string &name);

  rhi::IBuffer *GetBuffer(const std::string &name);

  rhi::IImage *GetImage(const std::string &name);

  void SetPublicResource(const std::string &name, rhi::ObjPtr<rhi::IShader> &&shader);
  void SetPublicResource(const std::string &name, rhi::ObjPtr<rhi::IComputeProgram> &&program);
  void SetPublicResource(const std::string &name, rhi::ObjPtr<rhi::IBuffer> &&buffer);
  void SetPublicResource(const std::string &name, rhi::ObjPtr<rhi::IImage> &&image);

 private:
  void LoadPublicShaders();
  void LoadPublicBuffers();
  void LoadPublicImages();

  rhi::IDevice *device_{nullptr};

  VirtualFileSystem shaders_vfs_;

  std::map<std::string, rhi::ObjPtr<rhi::IShader>> shaders_;
  std::map<std::string, rhi::ObjPtr<rhi::IComputeProgram>> compute_programs_;
  std::map<std::string, rhi::ObjPtr<rhi::IBuffer>> buffers_;
  std::map<std::string, rhi::ObjPtr<rhi::IImage>> images_;
};
}  // namespace sparkium
