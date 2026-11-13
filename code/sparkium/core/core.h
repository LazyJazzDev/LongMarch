#pragma once
#include "sparkium/core/core_util.h"
#include "sparkium/core/resource_updates.h"

namespace sparkium {
class Core : public Object {
 public:
  // Borrowed device; destroy scenes, resources and this Core before the device.
  static std::unique_ptr<Core> Create(rhi::IDevice *device);

  bool IsValid() const;
  bool ReportFailure(const char *message);
  bool QueueUpload(rhi::IBuffer *buffer, const void *data, uint64_t size, uint64_t offset = 0);
  bool QueueUpload(rhi::IImage *image, const void *data, uint64_t row_pitch = 0);
  bool QueueBuild(rhi::IBottomLevelAccelerationStructure *structure,
                  const std::vector<rhi::AccelerationStructureGeometryBuffers> &geometries);
  ResourceUpdates::Cursor BeginUpdates() const;
  bool RecordUpdates(rhi::ICommandContext *commands, ResourceUpdates::Cursor &cursor);
  bool AcceptUpdates(const ResourceUpdates::Cursor &cursor);

  rhi::IDevice *GetDevice() const;

  // Resolve automatic selection and supported fallbacks for rendering and UI display.
  RenderPipeline ResolveRenderPipeline(RenderPipeline render_pipeline) const;

  bool RecordRender(Renderer &renderer,
                    rhi::ICommandContext *commands,
                    ResourceUpdates::Cursor &updates,
                    Scene *scene,
                    Camera *camera,
                    Film *film,
                    RenderPipeline render_pipeline = RENDER_PIPELINE_AUTO);

  const VirtualFileSystem &GetShadersVFS() const;

  rhi::IShader *GetShader(const std::string &name);

  rhi::IComputeProgram *GetComputeProgram(const std::string &name);

  rhi::IBuffer *GetBuffer(const std::string &name);

  rhi::IImage *GetImage(const std::string &name);

  bool SetPublicResource(const std::string &name, rhi::ObjPtr<rhi::IShader> &&shader);
  bool SetPublicResource(const std::string &name, rhi::ObjPtr<rhi::IComputeProgram> &&program);
  bool SetPublicResource(const std::string &name, rhi::ObjPtr<rhi::IBuffer> &&buffer);
  bool SetPublicResource(const std::string &name, rhi::ObjPtr<rhi::IImage> &&image);

 private:
  explicit Core(rhi::IDevice *device);
  ResourceUpdates updates_;
  bool valid_{true};
  bool LoadPublicShaders();
  bool LoadPublicBuffers();
  bool LoadPublicImages();

  rhi::IDevice *device_{nullptr};

  VirtualFileSystem shaders_vfs_;

  std::map<std::string, rhi::ObjPtr<rhi::IShader>> shaders_;
  std::map<std::string, rhi::ObjPtr<rhi::IComputeProgram>> compute_programs_;
  std::map<std::string, rhi::ObjPtr<rhi::IBuffer>> buffers_;
  std::map<std::string, rhi::ObjPtr<rhi::IImage>> images_;
};
}  // namespace sparkium
