#include "sparkium/pipelines/raytracing/core/camera.h"

#include "sparkium/camera/cameras.h"
#include "sparkium/core/render_layout.h"
#include "sparkium/pipelines/raytracing/camera/camera_pinhole.h"
#include "sparkium/pipelines/raytracing/camera/camera_thin_lens.h"
#include "sparkium/pipelines/raytracing/core/core.h"

namespace sparkium::raytracing {

Camera::Camera(Core *core, const std::string &shader_file, const std::string &entry_point)
    : core_(core),
      shader_file_(shader_file),
      entry_point_(entry_point) {
}

rhi::IShader *Camera::Shader() {
  if (!core_->IsValid())
    return nullptr;
  if (!camera_shader_ && core_->GetDevice()->SupportRayTracingPipeline()) {
    auto shaders_vfs = core_->GetShadersVFS();
    camera_shader_ = core_->GetDevice()->CreateShader(shader_file_, entry_point_, rhi::SHADER_STAGE_CALLABLE,
                                                      &shaders_vfs, {"."}, {});
    if (!camera_shader_) {
      core_->ReportFailure("failed to compile camera shader");
      return nullptr;
    }
  }
  return camera_shader_.Get();
}

rhi::IBindingGroup *Camera::Bindings() {
  if (!bindings_)
    bindings_ = core_->GetDevice()->CreateBindingGroup(RayTracingLayout(false, 1, 1, 1).groups[SPARKIUM_GROUP_CAMERA]);
  return bindings_ && bindings_->BindResource(SPARKIUM_CAMERA_DATA, Buffer()) ? bindings_.Get() : nullptr;
}

const std::string &Camera::ShaderFile() const {
  return shader_file_;
}

const std::string &Camera::EntryPoint() const {
  return entry_point_;
}

Camera *DedicatedCast(sparkium::Camera *camera) {
  DEDICATED_CAST(camera, sparkium::CameraPinhole, CameraPinhole);
  DEDICATED_CAST(camera, sparkium::CameraThinLens, CameraThinLens);
  if (camera)
    camera->GetCore()->ReportFailure("unsupported ray tracing camera model");
  return nullptr;
}

}  // namespace sparkium::raytracing
