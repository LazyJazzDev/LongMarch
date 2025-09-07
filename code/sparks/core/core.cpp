#include "sparks/core/core.h"

#include "sparks/core/camera.h"
#include "sparks/core/entity.h"
#include "sparks/core/film.h"
#include "sparks/core/geometry.h"
#include "sparks/core/material.h"
#include "sparks/core/scene.h"

namespace sparks {
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

void Core::ConvertFilmToRawImage(const Film &film, graphics::Image *image) {
  std::unique_ptr<graphics::CommandContext> cmd_context;
  core_->CreateCommandContext(&cmd_context);
  cmd_context->CmdBindComputeProgram(compute_programs_["film2img"].get());
  cmd_context->CmdBindResources(0, {film.accumulated_color_.get()}, graphics::BIND_POINT_COMPUTE);
  cmd_context->CmdBindResources(1, {film.accumulated_samples_.get()}, graphics::BIND_POINT_COMPUTE);
  cmd_context->CmdBindResources(2, {image}, graphics::BIND_POINT_COMPUTE);
  cmd_context->CmdDispatch((image->Extent().width + 7) / 8, (image->Extent().height + 7) / 8, 1);
  core_->SubmitCommandContext(cmd_context.get());
  core_->WaitGPU();
}

void Core::ToneMapping(graphics::Image *raw_image, graphics::Image *output_image) {
  std::unique_ptr<graphics::CommandContext> cmd_context;
  core_->CreateCommandContext(&cmd_context);
  cmd_context->CmdBindComputeProgram(compute_programs_["tone_mapping"].get());
  cmd_context->CmdBindResources(0, {raw_image}, graphics::BIND_POINT_COMPUTE);
  cmd_context->CmdBindResources(1, {output_image}, graphics::BIND_POINT_COMPUTE);
  cmd_context->CmdDispatch((output_image->Extent().width + 7) / 8, (output_image->Extent().height + 7) / 8, 1);
  core_->SubmitCommandContext(cmd_context.get());
  core_->WaitGPU();
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

void Core::LoadPublicShaders() {
  shaders_vfs_ = VirtualFileSystem::LoadDirectory(LONGMARCH_SPARKS_SHADERS);
  core_->CreateShader(shaders_vfs_, "film2img.hlsl", "Main", "cs_6_0", &shaders_["film2img"]);
  auto &film2img_program = compute_programs_["film2img"];
  core_->CreateComputeProgram(shaders_["film2img"].get(), &film2img_program);
  film2img_program->AddResourceBinding(graphics::RESOURCE_TYPE_TEXTURE, 1);
  film2img_program->AddResourceBinding(graphics::RESOURCE_TYPE_TEXTURE, 1);
  film2img_program->AddResourceBinding(graphics::RESOURCE_TYPE_IMAGE, 1);
  film2img_program->Finalize();

  core_->CreateShader(shaders_vfs_, "tone_mapping.hlsl", "Main", "cs_6_0", &shaders_["tone_mapping"]);
  auto &tone_mapping_program = compute_programs_["tone_mapping"];
  core_->CreateComputeProgram(shaders_["tone_mapping"].get(), &tone_mapping_program);
  tone_mapping_program->AddResourceBinding(graphics::RESOURCE_TYPE_TEXTURE, 1);
  tone_mapping_program->AddResourceBinding(graphics::RESOURCE_TYPE_IMAGE, 1);
  tone_mapping_program->Finalize();

  core_->CreateShader(shaders_vfs_, "blelloch_scan.hlsl", "BlellochUpSweep", "cs_6_3", {"-I."},
                      &shaders_["blelloch_scan_up"]);
  core_->CreateShader(shaders_vfs_, "blelloch_scan.hlsl", "BlellochDownSweep", "cs_6_3", {"-I."},
                      &shaders_["blelloch_scan_down"]);
  core_->CreateComputeProgram(shaders_["blelloch_scan_up"].get(), &compute_programs_["blelloch_scan_up"]);
  core_->CreateComputeProgram(shaders_["blelloch_scan_down"].get(), &compute_programs_["blelloch_scan_down"]);
  compute_programs_["blelloch_scan_up"]->AddResourceBinding(graphics::RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER, 1);
  compute_programs_["blelloch_scan_up"]->AddResourceBinding(graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);
  compute_programs_["blelloch_scan_up"]->Finalize();
  compute_programs_["blelloch_scan_down"]->AddResourceBinding(graphics::RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER, 1);
  compute_programs_["blelloch_scan_down"]->AddResourceBinding(graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);
  compute_programs_["blelloch_scan_down"]->Finalize();

  auto vfs = shaders_vfs_;
  vfs.WriteFile("material_sampler.hlsli", CodeLines{shaders_vfs_, "material/lambertian/sampler.hlsl"});
  vfs.WriteFile("entity_chit.hlsl", CodeLines{shaders_vfs_, "geometry/mesh/hit_group.hlsl"});
  core_->CreateShader(vfs, "entity_chit.hlsl", "RenderClosestHit", "lib_6_5", {"-I."},
                      &shaders_["mesh_lambertian_chit"]);
  core_->CreateShader(vfs, "entity_chit.hlsl", "ShadowClosestHit", "lib_6_5", {"-I."},
                      &shaders_["mesh_lambertian_shadow_chit"]);

  vfs.WriteFile("material_sampler.hlsli", CodeLines{shaders_vfs_, "material/light/sampler.hlsl"});
  core_->CreateShader(vfs, "entity_chit.hlsl", "RenderClosestHit", "lib_6_5", {"-I."}, &shaders_["mesh_light_chit"]);
  core_->CreateShader(vfs, "entity_chit.hlsl", "ShadowClosestHit", "lib_6_5", {"-I."},
                      &shaders_["mesh_light_shadow_chit"]);

  vfs.WriteFile("material_sampler.hlsli", CodeLines{shaders_vfs_, "material/principled/sampler.hlsl"});
  core_->CreateShader(vfs, "entity_chit.hlsl", "RenderClosestHit", "lib_6_5", {"-I."},
                      &shaders_["mesh_principled_chit"]);
  core_->CreateShader(vfs, "entity_chit.hlsl", "ShadowClosestHit", "lib_6_5", {"-I."},
                      &shaders_["mesh_principled_shadow_chit"]);

  vfs.WriteFile("material_sampler.hlsli", CodeLines{shaders_vfs_, "material/specular/sampler.hlsl"});
  core_->CreateShader(vfs, "entity_chit.hlsl", "RenderClosestHit", "lib_6_5", {"-I."}, &shaders_["mesh_specular_chit"]);
  core_->CreateShader(vfs, "entity_chit.hlsl", "ShadowClosestHit", "lib_6_5", {"-I."},
                      &shaders_["mesh_specular_shadow_chit"]);
}

void Core::LoadPublicBuffers() {
  auto path = FindAssetFile("data/new-joe-kuo-7.21201");
  auto data = SobolTableGen(65536, 1024, path);
  core_->CreateBuffer(data.size() * sizeof(float), graphics::BUFFER_TYPE_STATIC, &buffers_["sobol"]);
  buffers_["sobol"]->UploadData(data.data(), data.size() * sizeof(float));
}

void Core::LoadPublicImages() {
  uint32_t white_pixel = 0xFFFFFFFF;
  float white_hdr_pixel[] = {1.0f, 1.0f, 1.0f, 1.0f};
  core_->CreateImage(1, 1, graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &images_["white"]);
  core_->CreateImage(1, 1, graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT, &images_["white_hdr"]);
  images_["white"]->UploadData(&white_pixel);
  images_["white_hdr"]->UploadData(white_hdr_pixel);
}

}  // namespace sparks
