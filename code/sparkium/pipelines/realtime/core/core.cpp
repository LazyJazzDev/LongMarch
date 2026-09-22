#include "sparkium/pipelines/realtime/core/core.h"

#include "sparkium/pipelines/realtime/core/camera.h"
#include "sparkium/pipelines/realtime/core/scene.h"

namespace sparkium::realtime {

Core::Core(sparkium::Core &core) : core_(core) {
  shaders_vfs_ = core_.CreatePipelineShadersVFS("realtime");
  LoadPublicShaders();
}

graphics::Core *Core::GraphicsCore() const {
  return core_.GraphicsCore();
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

graphics::Image *Core::GetImage(const std::string &name) {
  return core_.GetImage(name);
}

graphics::Buffer *Core::GetBuffer(const std::string &name) {
  return core_.GetBuffer(name);
}

void Core::LoadPublicShaders() {
  std::unique_ptr<graphics::Shader> shader;
  std::unique_ptr<graphics::ComputeProgram> compute_program;
  auto &shaders_vfs = shaders_vfs_;
  core_.GraphicsCore()->CreateShader(shaders_vfs, "blelloch_scan.hlsl", "BlellochUpSweep", "cs_6_3", {"-I."}, &shader);
  shaders_["blelloch_scan_up"] = std::move(shader);
  core_.GraphicsCore()->CreateShader(shaders_vfs, "blelloch_scan.hlsl", "BlellochDownSweep", "cs_6_3", {"-I."},
                                     &shader);
  shaders_["blelloch_scan_down"] = std::move(shader);

  core_.GraphicsCore()->CreateComputeProgram(GetShader("blelloch_scan_up"), &compute_program);
  compute_program->AddResourceBinding(graphics::RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER, 1);
  compute_program->AddResourceBinding(graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);
  compute_program->Finalize();
  compute_programs_["blelloch_scan_up"] = std::move(compute_program);

  core_.GraphicsCore()->CreateComputeProgram(GetShader("blelloch_scan_down"), &compute_program);
  compute_program->AddResourceBinding(graphics::RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER, 1);
  compute_program->AddResourceBinding(graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);
  compute_program->Finalize();
  compute_programs_["blelloch_scan_down"] = std::move(compute_program);
}

Core *DedicatedCast(sparkium::Core *core) {
  COMPONENT_CAST(core, Core)
  return nullptr;
}

}  // namespace sparkium::realtime
