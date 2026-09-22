#include "sparkium/core/core.h"

#include "sparkium/core/camera.h"
#include "sparkium/core/entity.h"
#include "sparkium/core/film.h"
#include "sparkium/core/geometry.h"
#include "sparkium/core/material.h"
#include "sparkium/core/scene.h"

namespace sparkium {
Core::Core(graphics::Core *core) : core_(core) {
  LoadPublicShaders();
  LoadPublicBuffers();
  LoadPublicImages();
}

graphics::Core *Core::GraphicsCore() const {
  return core_;
}

const VirtualFileSystem &Core::GetShadersVFS() const {
  return shaders_vfs_;
}

VirtualFileSystem Core::CreatePipelineShadersVFS(const std::string &pipeline) const {
  if (pipeline != "raytracing" && pipeline != "realtime")
    throw std::invalid_argument("unknown shader pipeline: " + pipeline);
  VirtualFileSystem result;
  const std::filesystem::path root = LONGMARCH_SPARKIUM_SHADERS;
  for (const auto &library : {std::string("common"), pipeline}) {
    const auto directory = root / library;
    for (const auto &entry : std::filesystem::recursive_directory_iterator(directory)) {
      if (!entry.is_regular_file() || (entry.path().extension() != ".hlsl" && entry.path().extension() != ".hlsli"))
        continue;
      const auto logical_path = std::filesystem::relative(entry.path(), directory).generic_string();
      std::vector<uint8_t> source;
      if (result.ReadFile(logical_path, source) == 0)
        throw std::runtime_error("duplicate shader include path: " + logical_path);
      if (shaders_vfs_.ReadFile(library + "/" + logical_path, source) != 0)
        throw std::runtime_error("missing shader source: " + library + "/" + logical_path);
      result.WriteFile(logical_path, source);
    }
  }
  return result;
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

  for (bool hdr : {false, true}) {
    const std::string name = hdr ? "tone_mapping_hdr" : "tone_mapping";
    std::vector<std::string> args;
    if (hdr)
      args.push_back("-DSPARKIUM_HDR_OUTPUT=1");
    core_->CreateShader(shaders_vfs_, "common/tone_mapping.hlsl", "Main", "cs_6_0", args, &shader);
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
  std::unique_ptr<graphics::Buffer> buffer;
  core_->CreateBuffer(data.size() * sizeof(float), graphics::BUFFER_TYPE_STATIC, &buffer);
  buffer->UploadData(data.data(), data.size() * sizeof(float));
  SetPublicResource("sobol", std::move(buffer));
}

void Core::LoadPublicImages() {
  uint32_t pixel = 0xFFFFFFFF;
  float hdr_pixel[] = {1.0f, 1.0f, 1.0f, 1.0f};

  std::unique_ptr<graphics::Image> image;
  core_->CreateImage(1, 1, graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &image);
  image->UploadData(&pixel);
  SetPublicResource("white", std::move(image));

  core_->CreateImage(1, 1, graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT, &image);
  image->UploadData(hdr_pixel);
  SetPublicResource("white_hdr", std::move(image));

  pixel = 0;
  hdr_pixel[0] = 0.0f;
  hdr_pixel[1] = 0.0f;
  hdr_pixel[2] = 0.0f;
  hdr_pixel[3] = 1.0f;

  core_->CreateImage(1, 1, graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &image);
  image->UploadData(&pixel);
  SetPublicResource("black", std::move(image));

  core_->CreateImage(1, 1, graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT, &image);
  image->UploadData(hdr_pixel);
  SetPublicResource("black_hdr", std::move(image));

  pixel = 0xFFFF8080;  // Normal map default value (0.5, 0.5, 1.0)
  core_->CreateImage(1, 1, graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &image);
  image->UploadData(&pixel);
  SetPublicResource("normal_default", std::move(image));
}

}  // namespace sparkium
