#include "sparkium/backend/cuda/path_tracing/core/core.h"

#include "sparkium/backend/cuda/path_tracing/core/camera.h"
#include "sparkium/backend/cuda/path_tracing/core/film.h"
#include "sparkium/backend/cuda/path_tracing/core/scene.h"

namespace sparkium::cuda_tracing {

Core::Core(sparkium::Core &core) : core_(core) {
  LoadPublicShaders();
}

backend::Device *Core::BackendDevice() const {
  return core_.BackendDevice();
}

const VirtualFileSystem &Core::GetShadersVFS() const {
  return core_.GetShadersVFS();
}

graphics::Shader *Core::GetShader(const std::string &name) {
  return core_.GetShader(name);
}

graphics::ComputeProgram *Core::GetComputeProgram(const std::string &name) {
  return core_.GetComputeProgram(name);
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
  auto &shaders_vfs = core_.GetShadersVFS();
  core_.BackendDevice()->CreateShader(shaders_vfs, "film2img.hlsl", "Main", "cs_6_0", &shader);
  core_.SetPublicResource("film2img", std::move(shader));

  core_.BackendDevice()->CreateComputeProgram(core_.GetShader("film2img"), &compute_program);
  compute_program->AddResourceBinding(graphics::RESOURCE_TYPE_IMAGE, 1);
  compute_program->AddResourceBinding(graphics::RESOURCE_TYPE_IMAGE, 1);
  compute_program->AddResourceBinding(graphics::RESOURCE_TYPE_WRITABLE_IMAGE, 1);
  compute_program->Finalize();
  core_.SetPublicResource("film2img", std::move(compute_program));

  core_.BackendDevice()->CreateShader(shaders_vfs, "blelloch_scan.hlsl", "BlellochUpSweep", "cs_6_3", {"-I."}, &shader);
  core_.SetPublicResource("blelloch_scan_up", std::move(shader));
  core_.BackendDevice()->CreateShader(shaders_vfs, "blelloch_scan.hlsl", "BlellochDownSweep", "cs_6_3", {"-I."},
                                      &shader);
  core_.SetPublicResource("blelloch_scan_down", std::move(shader));

  core_.BackendDevice()->CreateComputeProgram(core_.GetShader("blelloch_scan_up"), &compute_program);
  compute_program->AddResourceBinding(graphics::RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER, 1);
  compute_program->AddResourceBinding(graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);
  compute_program->Finalize();
  core_.SetPublicResource("blelloch_scan_up", std::move(compute_program));

  core_.BackendDevice()->CreateComputeProgram(core_.GetShader("blelloch_scan_down"), &compute_program);
  compute_program->AddResourceBinding(graphics::RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER, 1);
  compute_program->AddResourceBinding(graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);
  compute_program->Finalize();
  core_.SetPublicResource("blelloch_scan_down", std::move(compute_program));
}

Core *DedicatedCast(sparkium::Core *core) {
  COMPONENT_CAST(core, Core)
  return nullptr;
}

void Render(sparkium::Core *core, sparkium::Scene *scene, sparkium::Camera *camera, sparkium::Film *film, bool optix) {
  auto rt_core = DedicatedCast(core);
  auto rt_scene = DedicatedCast(scene);
  auto rt_film = DedicatedCast(film);
  auto rt_camera = DedicatedCast(camera);
  rt_scene->Render(rt_camera, rt_film, optix);
}

}  // namespace sparkium::cuda_tracing
