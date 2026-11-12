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
}

std::unique_ptr<Core> Core::Create(rhi::IDevice *device) {
  if (!device)
    return {};
  std::unique_ptr<Core> core(new Core(device));
  if (!core->LoadPublicShaders() || !core->LoadPublicBuffers() || !core->LoadPublicImages())
    return {};
  return core;
}

bool Core::IsValid() const {
  return valid_;
}

bool Core::ReportFailure(const char *message) {
  LogError("Sparkium: {}", message);
  valid_ = false;
  return false;
}

bool Core::QueueUpload(rhi::IBuffer *buffer, const void *data, uint64_t size, uint64_t offset) {
  return (valid_ && updates_.Upload(buffer, data, size, offset)) || ReportFailure("Cannot queue buffer upload");
}

bool Core::QueueUpload(rhi::IImage *image, const void *data, uint64_t row_pitch) {
  return (valid_ && updates_.Upload(image, data, row_pitch)) || ReportFailure("Cannot queue image upload");
}

bool Core::QueueBuild(rhi::IBottomLevelAccelerationStructure *structure,
                      const std::vector<rhi::AccelerationStructureGeometryBuffers> &geometries) {
  return (valid_ && updates_.Build(structure, geometries)) || ReportFailure("Cannot queue geometry build");
}

ResourceUpdates::Cursor Core::BeginUpdates() const {
  return updates_.Begin();
}

bool Core::RecordUpdates(rhi::ICommandContext *commands, ResourceUpdates::Cursor &cursor) {
  return valid_ && updates_.Record(commands, cursor);
}

bool Core::AcceptUpdates(const ResourceUpdates::Cursor &cursor) {
  return updates_.Accept(cursor);
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

bool Core::RecordRender(Renderer &renderer,
                        rhi::ICommandContext *commands,
                        ResourceUpdates::Cursor &updates,
                        Scene *scene,
                        Camera *camera,
                        Film *film,
                        RenderPipeline render_pipeline) {
  if (!valid_ || !commands || !scene || !camera || !film)
    return false;
  switch (ResolveRenderPipeline(render_pipeline)) {
    case RENDER_PIPELINE_RASTERIZATION:
      return raster::RecordRender(this, commands, updates, scene, camera, film);
    case RENDER_PIPELINE_RAY_TRACING:
      return raytracing::RecordRender(this, renderer, commands, updates, scene, camera, film);
    case RENDER_PIPELINE_RAY_QUERY:
      return device_->SupportRayQuery() &&
             raytracing::RecordRender(this, renderer, commands, updates, scene, camera, film, true, true);
    case RENDER_PIPELINE_RT_FALLBACK:
      return raytracing::RecordRender(this, renderer, commands, updates, scene, camera, film, true);
    default:
      return false;
  }
}

const VirtualFileSystem &Core::GetShadersVFS() const {
  return shaders_vfs_;
}

rhi::IShader *Core::GetShader(const std::string &name) {
  auto found = shaders_.find(name);
  return found == shaders_.end() ? nullptr : found->second.Get();
}

rhi::IComputeProgram *Core::GetComputeProgram(const std::string &name) {
  auto found = compute_programs_.find(name);
  return found == compute_programs_.end() ? nullptr : found->second.Get();
}

rhi::IBuffer *Core::GetBuffer(const std::string &name) {
  auto found = buffers_.find(name);
  return found == buffers_.end() ? nullptr : found->second.Get();
}

rhi::IImage *Core::GetImage(const std::string &name) {
  auto found = images_.find(name);
  return found == images_.end() ? nullptr : found->second.Get();
}

bool Core::SetPublicResource(const std::string &name, rhi::ObjPtr<rhi::IShader> &&shader) {
  if (!shader)
    return ReportFailure("Cannot create public resource");
  shaders_[name] = std::move(shader);
  return true;
}

bool Core::SetPublicResource(const std::string &name, rhi::ObjPtr<rhi::IComputeProgram> &&program) {
  if (!program)
    return ReportFailure("Cannot create public resource");
  compute_programs_[name] = std::move(program);
  return true;
}

bool Core::SetPublicResource(const std::string &name, rhi::ObjPtr<rhi::IBuffer> &&buffer) {
  if (!buffer)
    return ReportFailure("Cannot create public resource");
  buffers_[name] = std::move(buffer);
  return true;
}

bool Core::SetPublicResource(const std::string &name, rhi::ObjPtr<rhi::IImage> &&image) {
  if (!image)
    return ReportFailure("Cannot create public resource");
  images_[name] = std::move(image);
  return true;
}

bool Core::LoadPublicShaders() {
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
    if (!SetPublicResource(name, std::move(shader)))
      return false;
    {
      rhi::ComputeProgramDesc desc;
      desc.compute_shader = GetShader(name);
      desc.pipeline_layout.groups = {
          {{{0, rhi::BINDING_TEXTURE2D, 1}, {1, rhi::BINDING_RW_TEXTURE2D, 1}, {2, rhi::BINDING_UNIFORM_BUFFER, 1}}}};
      compute_program = device_->CreateComputeProgram(desc);
    }
    if (!SetPublicResource(name, std::move(compute_program)))
      return false;
  }
  return true;
}

bool Core::LoadPublicBuffers() {
  auto path = FindAssetFile("data/new-joe-kuo-7.21201");
  auto data = SobolTableGen(65536, 1024, path);
  rhi::ObjPtr<rhi::IBuffer> buffer;
  buffer = device_->CreateBuffer({data.size() * sizeof(float)});
  QueueUpload(buffer.Get(), data.data(), data.size() * sizeof(float));
  return SetPublicResource("sobol", std::move(buffer)) && valid_;
}

bool Core::LoadPublicImages() {
  uint32_t pixel = 0xFFFFFFFF;
  float hdr_pixel[] = {1.0f, 1.0f, 1.0f, 1.0f};

  rhi::ObjPtr<rhi::IImage> image;
  image = device_->CreateImage({1, 1, rhi::IMAGE_FORMAT_R8G8B8A8_UNORM});
  QueueUpload(image.Get(), &pixel);
  SetPublicResource("white", std::move(image));

  image = device_->CreateImage({1, 1, rhi::IMAGE_FORMAT_R32G32B32A32_SFLOAT});
  QueueUpload(image.Get(), hdr_pixel);
  SetPublicResource("white_hdr", std::move(image));

  pixel = 0;
  hdr_pixel[0] = 0.0f;
  hdr_pixel[1] = 0.0f;
  hdr_pixel[2] = 0.0f;
  hdr_pixel[3] = 1.0f;

  image = device_->CreateImage({1, 1, rhi::IMAGE_FORMAT_R8G8B8A8_UNORM});
  QueueUpload(image.Get(), &pixel);
  SetPublicResource("black", std::move(image));

  image = device_->CreateImage({1, 1, rhi::IMAGE_FORMAT_R32G32B32A32_SFLOAT});
  QueueUpload(image.Get(), hdr_pixel);
  SetPublicResource("black_hdr", std::move(image));

  pixel = 0xFFFF8080;  // Normal map default value (0.5, 0.5, 1.0)
  image = device_->CreateImage({1, 1, rhi::IMAGE_FORMAT_R8G8B8A8_UNORM});
  QueueUpload(image.Get(), &pixel);
  return SetPublicResource("normal_default", std::move(image)) && valid_;
}

}  // namespace sparkium
