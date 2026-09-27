#include "sparkium/core/core.h"

#include "sparkium/core/camera.h"
#include "sparkium/core/entity.h"
#include "sparkium/core/film.h"
#include "sparkium/core/geometry.h"
#include "sparkium/core/material.h"
#include "sparkium/core/scene.h"
#include "sparkium/pipelines/pipelines.h"

namespace sparkium {
Core::Core(graphics::Core *core) : core_(core), data_updates_(core) {
  LoadPublicShaders();
  LoadPublicBuffers();
  LoadPublicImages();
}

int Core::CreateBuffer(size_t size, graphics::BufferType type, double_ptr<sparkium::Buffer> buffer) {
  if (type != graphics::BUFFER_TYPE_STATIC)
    throw std::invalid_argument("tracked buffers must use static GPU storage");
  std::unique_ptr<graphics::Buffer> native;
  int status = core_->CreateBuffer(size, type, &native);
  if (!status)
    buffer.construct(data_updates_, std::move(native));
  return status;
}

int Core::CreateImage(int width, int height, graphics::ImageFormat format, double_ptr<sparkium::Image> image) {
  std::unique_ptr<graphics::Image> native;
  int status = core_->CreateImage(width, height, format, &native);
  if (!status)
    image.construct(data_updates_, std::move(native));
  return status;
}

int Core::LoadImageFromFile(const std::string &path, double_ptr<sparkium::Image> image) {
  std::unique_ptr<graphics::Image> native;
  std::unique_ptr<sparkium::Image> loaded;
  int status = graphics::LoadImageFromFile(core_, path, &native, [&](graphics::Image *, const void *data) {
    loaded = std::make_unique<sparkium::Image>(data_updates_, std::move(native));
    loaded->Update(data);
  });
  if (!status)
    image = loaded.release();
  return status;
}

int Core::CreateBottomLevelAccelerationStructure(graphics::BufferRange vertices,
                                                 graphics::BufferRange indices,
                                                 uint32_t vertex_count,
                                                 uint32_t stride,
                                                 uint32_t primitive_count,
                                                 graphics::RayTracingGeometryFlag flags,
                                                 double_ptr<BottomLevelAccelerationStructure> blas) {
  blas.construct(data_updates_, vertices, indices, vertex_count, stride, primitive_count, flags);
  return 0;
}

int Core::CreateTopLevelAccelerationStructure(const std::vector<AccelerationStructureInstance> &instances,
                                              double_ptr<TopLevelAccelerationStructure> tlas) {
  tlas.construct(data_updates_, instances);
  return 0;
}

graphics::Core *Core::GraphicsCore() const {
  return core_;
}

RenderPipeline Core::ResolveRenderPipeline(RenderPipeline render_pipeline) const {
  if (render_pipeline == RENDER_PIPELINE_AUTO) {
    const bool prefer_ray_query =
        core_->API() == graphics::BACKEND_API_D3D12 || core_->API() == graphics::BACKEND_API_VULKAN;
    if (prefer_ray_query && core_->DeviceRayQuerySupport()) {
      render_pipeline = RENDER_PIPELINE_RAY_QUERY;
    } else if (core_->DeviceRayTracingSupport()) {
      render_pipeline = RENDER_PIPELINE_RAY_TRACING;
    } else if (core_->DeviceRayQuerySupport()) {
      render_pipeline = RENDER_PIPELINE_RAY_QUERY;
    } else {
      render_pipeline = RENDER_PIPELINE_RT_FALLBACK;
    }
  }
  // Older Blender scenes request pipeline RT. Keep them on native traversal
  // when the device supports inline queries instead of a full RT pipeline.
  if (render_pipeline == RENDER_PIPELINE_RAY_TRACING && !core_->DeviceRayTracingSupport())
    return core_->DeviceRayQuerySupport() ? RENDER_PIPELINE_RAY_QUERY : RENDER_PIPELINE_RT_FALLBACK;
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
      if (!core_->DeviceRayQuerySupport())
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

graphics::Shader *Core::GetShader(const std::string &name) {
  return shaders_[name].get();
}

graphics::ComputeProgram *Core::GetComputeProgram(const std::string &name) {
  return compute_programs_[name].get();
}

graphics::Buffer *Core::GetBuffer(const std::string &name) {
  auto it = buffers_.find(name);
  return it == buffers_.end() || !it->second ? nullptr : it->second->Get();
}

graphics::Image *Core::GetImage(const std::string &name) {
  auto it = images_.find(name);
  return it == images_.end() || !it->second ? nullptr : it->second->Get();
}

void Core::SetPublicResource(const std::string &name, std::unique_ptr<graphics::Shader> &&shader) {
  shaders_[name] = std::move(shader);
}

void Core::SetPublicResource(const std::string &name, std::unique_ptr<graphics::ComputeProgram> &&program) {
  compute_programs_[name] = std::move(program);
}

void Core::SetPublicResource(const std::string &name, std::unique_ptr<sparkium::Buffer> &&buffer) {
  buffers_[name] = std::move(buffer);
}

void Core::SetPublicResource(const std::string &name, std::unique_ptr<sparkium::Image> &&image) {
  images_[name] = std::move(image);
}

void Core::LoadPublicShaders() {
  shaders_vfs_ = VirtualFileSystem::LoadDirectory(LONGMARCH_SPARKIUM_SHADERS);
  std::unique_ptr<graphics::Shader> shader;
  std::unique_ptr<graphics::ComputeProgram> compute_program;

  for (bool hdr : {false, true}) {
    const std::string name = hdr ? "tone_mapping_hdr" : "tone_mapping";
    std::vector<std::string> args;
    if (hdr)
      args.push_back("-DSPARKIUM_HDR_OUTPUT=1");
    core_->CreateShader(shaders_vfs_, "tone_mapping.slang", "Main", "cs_6_0", args, &shader);
    SetPublicResource(name, std::move(shader));
    core_->CreateComputeProgram(GetShader(name), &compute_program);
    compute_program->AddResourceBinding(graphics::RESOURCE_TYPE_IMAGE, 1);
    compute_program->AddResourceBinding(graphics::RESOURCE_TYPE_WRITABLE_IMAGE, 1);
    compute_program->AddResourceBinding(graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);
    compute_program->Finalize();
    SetPublicResource(name, std::move(compute_program));
  }
}

void Core::LoadPublicBuffers() {
  auto path = FindAssetFile("data/new-joe-kuo-7.21201");
  auto data = SobolTableGen(65536, 1024, path);
  std::unique_ptr<sparkium::Buffer> buffer;
  CreateBuffer(data.size() * sizeof(float), graphics::BUFFER_TYPE_STATIC, &buffer);
  buffer->Update(data.data(), data.size() * sizeof(float));
  SetPublicResource("sobol", std::move(buffer));
}

void Core::LoadPublicImages() {
  uint32_t pixel = 0xFFFFFFFF;
  float hdr_pixel[] = {1.0f, 1.0f, 1.0f, 1.0f};

  std::unique_ptr<sparkium::Image> image;
  CreateImage(1, 1, graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &image);
  image->Update(&pixel);
  SetPublicResource("white", std::move(image));

  CreateImage(1, 1, graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT, &image);
  image->Update(hdr_pixel);
  SetPublicResource("white_hdr", std::move(image));

  pixel = 0;
  hdr_pixel[0] = 0.0f;
  hdr_pixel[1] = 0.0f;
  hdr_pixel[2] = 0.0f;
  hdr_pixel[3] = 1.0f;

  CreateImage(1, 1, graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &image);
  image->Update(&pixel);
  SetPublicResource("black", std::move(image));

  CreateImage(1, 1, graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT, &image);
  image->Update(hdr_pixel);
  SetPublicResource("black_hdr", std::move(image));

  pixel = 0xFFFF8080;  // Normal map default value (0.5, 0.5, 1.0)
  CreateImage(1, 1, graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &image);
  image->Update(&pixel);
  SetPublicResource("normal_default", std::move(image));
}

}  // namespace sparkium
