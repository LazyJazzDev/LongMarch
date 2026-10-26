#include "sparkium/core/core.h"

#include "sparkium/core/camera.h"
#include "sparkium/core/entity.h"
#include "sparkium/core/film.h"
#include "sparkium/core/geometry.h"
#include "sparkium/core/material.h"
#include "sparkium/core/scene.h"
#include "sparkium/pipelines/pipelines.h"

namespace sparkium {
Core::Core(rhi::IDevice *device) : device_(device) {
  LoadPublicShaders();
  LoadPublicBuffers();
  LoadPublicImages();
}

rhi::IDevice *Core::GetDevice() const {
  return device_;
}

RenderPipeline Core::ResolveRenderPipeline(RenderPipeline render_pipeline) const {
  if (render_pipeline == RENDER_PIPELINE_AUTO) {
    if (device_->SupportRayQuery()) {
      render_pipeline = RENDER_PIPELINE_RAY_QUERY;
    } else if (device_->SupportRayTracingPipeline()) {
      render_pipeline = RENDER_PIPELINE_RAY_TRACING;
    } else {
      render_pipeline = RENDER_PIPELINE_RT_FALLBACK;
    }
  }
  // Older Blender scenes request pipeline RT. Keep them on native traversal
  // when the device supports inline queries instead of a full RT pipeline.
  if (render_pipeline == RENDER_PIPELINE_RAY_TRACING && !device_->SupportRayTracingPipeline())
    return device_->SupportRayQuery() ? RENDER_PIPELINE_RAY_QUERY : RENDER_PIPELINE_RT_FALLBACK;
  return render_pipeline;
}

void Core::Render(Scene *scene, Camera *camera, Film *film, RenderPipeline render_pipeline) {
  render_pipeline = ResolveRenderPipeline(render_pipeline);
  switch (render_pipeline) {
    case RENDER_PIPELINE_RASTERIZATION:
      raster::Render(this, scene, camera, film);
      break;
    case RENDER_PIPELINE_RAY_TRACING:
      raytracing::Render(this, scene, camera, film);
      break;
    case RENDER_PIPELINE_RAY_QUERY:
      if (!device_->SupportRayQuery())
        throw std::runtime_error("ray_query is unavailable on the selected graphics backend");
      raytracing::Render(this, scene, camera, film, true, true);
      break;
    case RENDER_PIPELINE_RT_FALLBACK:
      raytracing::Render(this, scene, camera, film, true);
      break;
    default:
      LogError("Unknown render pipeline");
  }
}

const VirtualFileSystem &Core::GetShadersVFS() const {
  return shaders_vfs_;
}

rhi::IShader *Core::GetShader(const std::string &name) {
  return shaders_[name].Get();
}

rhi::IComputeProgram *Core::GetComputeProgram(const std::string &name) {
  return compute_programs_[name].Get();
}

rhi::IBuffer *Core::GetBuffer(const std::string &name) {
  return buffers_[name].Get();
}

rhi::IImage *Core::GetImage(const std::string &name) {
  return images_[name].Get();
}

void Core::SetPublicResource(const std::string &name, rhi::ObjPtr<rhi::IShader> &&shader) {
  shaders_[name] = std::move(shader);
}

void Core::SetPublicResource(const std::string &name, rhi::ObjPtr<rhi::IComputeProgram> &&program) {
  compute_programs_[name] = std::move(program);
}

void Core::SetPublicResource(const std::string &name, rhi::ObjPtr<rhi::IBuffer> &&buffer) {
  buffers_[name] = std::move(buffer);
}

void Core::SetPublicResource(const std::string &name, rhi::ObjPtr<rhi::IImage> &&image) {
  images_[name] = std::move(image);
}

void Core::LoadPublicShaders() {
  shaders_vfs_ = VirtualFileSystem::LoadDirectory(LONGMARCH_SPARKIUM_SHADERS);
  rhi::ObjPtr<rhi::IShader> shader;
  rhi::ObjPtr<rhi::IComputeProgram> compute_program;

  for (bool hdr : {false, true}) {
    const std::string name = hdr ? "tone_mapping_hdr" : "tone_mapping";
    std::vector<std::pair<std::string, std::string>> defines;
    if (hdr)
      defines.emplace_back("SPARKIUM_HDR_OUTPUT", "1");
    shader =
        device_->CreateShader("tone_mapping.slang", "Main", rhi::SHADER_STAGE_COMPUTE, &shaders_vfs_, {"."}, defines);
    SetPublicResource(name, std::move(shader));
    {
      rhi::ComputeProgramDesc desc;
      desc.compute_shader = GetShader(name);
      desc.pipeline_layout.groups = {
          {{{0, rhi::BINDING_TEXTURE2D, 1}, {1, rhi::BINDING_RW_TEXTURE2D, 1}, {2, rhi::BINDING_UNIFORM_BUFFER, 1}}}};
      compute_program = device_->CreateComputeProgram(desc);
    }
    SetPublicResource(name, std::move(compute_program));
  }
}

void Core::LoadPublicBuffers() {
  auto path = FindAssetFile("data/new-joe-kuo-7.21201");
  auto data = SobolTableGen(65536, 1024, path);
  rhi::ObjPtr<rhi::IBuffer> buffer;
  buffer = device_->CreateBuffer({data.size() * sizeof(float)});
  UploadBuffer(device_, buffer.Get(), data.data(), data.size() * sizeof(float));
  SetPublicResource("sobol", std::move(buffer));
}

void Core::LoadPublicImages() {
  uint32_t pixel = 0xFFFFFFFF;
  float hdr_pixel[] = {1.0f, 1.0f, 1.0f, 1.0f};

  rhi::ObjPtr<rhi::IImage> image;
  image = device_->CreateImage({1, 1, rhi::IMAGE_FORMAT_R8G8B8A8_UNORM});
  UploadImage(device_, image.Get(), &pixel);
  SetPublicResource("white", std::move(image));

  image = device_->CreateImage({1, 1, rhi::IMAGE_FORMAT_R32G32B32A32_SFLOAT});
  UploadImage(device_, image.Get(), hdr_pixel);
  SetPublicResource("white_hdr", std::move(image));

  pixel = 0;
  hdr_pixel[0] = 0.0f;
  hdr_pixel[1] = 0.0f;
  hdr_pixel[2] = 0.0f;
  hdr_pixel[3] = 1.0f;

  image = device_->CreateImage({1, 1, rhi::IMAGE_FORMAT_R8G8B8A8_UNORM});
  UploadImage(device_, image.Get(), &pixel);
  SetPublicResource("black", std::move(image));

  image = device_->CreateImage({1, 1, rhi::IMAGE_FORMAT_R32G32B32A32_SFLOAT});
  UploadImage(device_, image.Get(), hdr_pixel);
  SetPublicResource("black_hdr", std::move(image));

  pixel = 0xFFFF8080;  // Normal map default value (0.5, 0.5, 1.0)
  image = device_->CreateImage({1, 1, rhi::IMAGE_FORMAT_R8G8B8A8_UNORM});
  UploadImage(device_, image.Get(), &pixel);
  SetPublicResource("normal_default", std::move(image));
}

}  // namespace sparkium
