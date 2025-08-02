#include "sparks/core/core.h"

#include "sparks/core/camera.h"
#include "sparks/core/entity.h"
#include "sparks/core/film.h"
#include "sparks/core/geometry.h"
#include "sparks/core/material.h"
#include "sparks/core/scene.h"

namespace sparks {
Core::Core(graphics::Core *core) : core_(core) {
  shaders_vfs_ = VirtualFileSystem::LoadDirectory(LONGMARCH_SPARKS_SHADERS);
  core_->CreateShader(shaders_vfs_, "film2img.hlsl", "Main", "cs_6_0", &film2img_shader_);
  core_->CreateComputeProgram(film2img_shader_.get(), &film2img_program_);
  film2img_program_->AddResourceBinding(graphics::RESOURCE_TYPE_TEXTURE, 1);
  film2img_program_->AddResourceBinding(graphics::RESOURCE_TYPE_TEXTURE, 1);
  film2img_program_->AddResourceBinding(graphics::RESOURCE_TYPE_IMAGE, 1);
  film2img_program_->Finalize();
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
}

graphics::Core *Core::GraphicsCore() const {
  return core_;
}

const VirtualFileSystem &Core::GetShadersVFS() const {
  return shaders_vfs_;
}

void Core::ConvertFilmToImage(const Film &film, graphics::Image *image) {
  std::unique_ptr<graphics::CommandContext> cmd_context;
  core_->CreateCommandContext(&cmd_context);
  cmd_context->CmdBindComputeProgram(film2img_program_.get());
  cmd_context->CmdBindResources(0, {film.accumulated_color_.get()}, graphics::BIND_POINT_COMPUTE);
  cmd_context->CmdBindResources(1, {film.accumulated_samples_.get()}, graphics::BIND_POINT_COMPUTE);
  cmd_context->CmdBindResources(2, {image}, graphics::BIND_POINT_COMPUTE);
  cmd_context->CmdDispatch((image->Extent().width + 7) / 8, (image->Extent().height + 7) / 8, 1);
  core_->SubmitCommandContext(cmd_context.get());
  core_->WaitGPU();
}

graphics::Shader *Core::GetShader(const std::string &name) {
  return shaders_[name].get();
}

graphics::ComputeProgram *Core::GetComputeProgram(const std::string &name) {
  return compute_programs_[name].get();
}

}  // namespace sparks
