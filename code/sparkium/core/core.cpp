#include "sparkium/core/core.h"

#include "sparkium/core/camera.h"
#include "sparkium/core/entity.h"
#include "sparkium/core/film.h"
#include "sparkium/core/geometry.h"
#include "sparkium/core/material.h"
#include "sparkium/core/scene.h"
#include "sparkium/pipelines/pipelines.h"

namespace sparkium {
Core::Core(graphics::Core *core) : core_(core) {
  LoadPublicShaders();
  LoadPublicBuffers();
  LoadPublicImages();
}

graphics::Core *Core::GraphicsCore() const {
  return core_;
}

void Core::Render(Scene *scene, Camera *camera, Film *film, RenderPipeline render_pipeline) {
  if (render_pipeline == RENDER_PIPELINE_AUTO) {
    if (core_->DeviceRayTracingSupport()) {
      render_pipeline = RENDER_PIPELINE_RAY_TRACING;
    } else {
      render_pipeline = RENDER_PIPELINE_RASTERIZATION;
    }
  }
  switch (render_pipeline) {
    case RENDER_PIPELINE_RASTERIZATION:
      raster::Render(this, scene, camera, film);
      break;
    case RENDER_PIPELINE_RAY_TRACING:
      if (!core_->DeviceRayTracingSupport()) {
        LogError("Ray tracing not supported on this device");
        return;
      }
      raytracing::Render(this, scene, camera, film);
      break;
    default:
      LogError("Unknown render pipeline");
  }
}

const VirtualFileSystem &Core::GetShadersVFS() const {
  return shaders_vfs_;
}

graphics::Shader *Core::GetShader(const std::string &name) {
  return shaders_[name].get();
}

graphics::ComputeProgram *Core::GetComputeProgram(const std::string &name) {
  return compute_programs_[name].get();
}

graphics::Buffer *Core::GetBuffer(const std::string &name) {
  return buffers_[name].get();
}

graphics::Image *Core::GetImage(const std::string &name) {
  return images_[name].get();
}

void Core::SetPublicResource(const std::string &name, std::unique_ptr<graphics::Shader> &&shader) {
  shaders_[name] = std::move(shader);
}

void Core::SetPublicResource(const std::string &name, std::unique_ptr<graphics::ComputeProgram> &&program) {
  compute_programs_[name] = std::move(program);
}

void Core::SetPublicResource(const std::string &name, std::unique_ptr<graphics::Buffer> &&buffer) {
  buffers_[name] = std::move(buffer);
}

void Core::SetPublicResource(const std::string &name, std::unique_ptr<graphics::Image> &&image) {
  images_[name] = std::move(image);
}

void Core::LoadPublicShaders() {
  shaders_vfs_ = VirtualFileSystem::LoadDirectory(LONGMARCH_SPARKIUM_SHADERS);
  std::unique_ptr<graphics::Shader> shader;
  std::unique_ptr<graphics::ComputeProgram> compute_program;

  core_->CreateShader(shaders_vfs_, "tone_mapping.hlsl", "Main", "cs_6_0", &shader);
  SetPublicResource("tone_mapping", std::move(shader));

  core_->CreateComputeProgram(GetShader("tone_mapping"), &compute_program);
  compute_program->AddResourceBinding(graphics::RESOURCE_TYPE_IMAGE, 1);
  compute_program->AddResourceBinding(graphics::RESOURCE_TYPE_WRITABLE_IMAGE, 1);
  compute_program->Finalize();
  SetPublicResource("tone_mapping", std::move(compute_program));
}

void Core::LoadPublicBuffers() {
  auto path = FindAssetFile("data/new-joe-kuo-7.21201");
  auto data = SobolTableGen(65536, 1024, path);
  std::unique_ptr<graphics::Buffer> buffer;
  core_->CreateBuffer(data.size() * sizeof(float), graphics::BUFFER_TYPE_STATIC, &buffer);
  buffer->UploadData(data.data(), data.size() * sizeof(float));
  SetPublicResource("sobol", std::move(buffer));
}

void Core::LoadPublicImages() {
  uint32_t white_pixel = 0xFFFFFFFF;
  float white_hdr_pixel[] = {1.0f, 1.0f, 1.0f, 1.0f};

  std::unique_ptr<graphics::Image> image;
  core_->CreateImage(1, 1, graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &image);
  image->UploadData(&white_pixel);
  SetPublicResource("white", std::move(image));

  core_->CreateImage(1, 1, graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT, &image);
  image->UploadData(white_hdr_pixel);
  SetPublicResource("white_hdr", std::move(image));
}

}  // namespace sparkium
