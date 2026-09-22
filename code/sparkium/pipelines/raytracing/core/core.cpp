#include "sparkium/pipelines/raytracing/core/core.h"

#include "sparkium/pipelines/raytracing/core/camera.h"
#include "sparkium/pipelines/raytracing/core/film.h"
#include "sparkium/pipelines/raytracing/core/scene.h"

namespace sparkium::raytracing {

Core::Core(sparkium::Core &core) : core_(core) {
  shaders_vfs_ = VirtualFileSystem::LoadDirectory(LONGMARCH_RAYTRACING_SHADERS);
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
  core_.GraphicsCore()->CreateShader(shaders_vfs, "film2img.hlsl", "Main", "cs_6_0", &shader);
  shaders_["film2img"] = std::move(shader);

  core_.GraphicsCore()->CreateComputeProgram(GetShader("film2img"), &compute_program);
  compute_program->AddResourceBinding(graphics::RESOURCE_TYPE_IMAGE, 1);
  compute_program->AddResourceBinding(graphics::RESOURCE_TYPE_IMAGE, 1);
  compute_program->AddResourceBinding(graphics::RESOURCE_TYPE_WRITABLE_IMAGE, 1);
  compute_program->Finalize();
  compute_programs_["film2img"] = std::move(compute_program);

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

  if (!core_.GraphicsCore()->DeviceRayTracingSupport())
    return;

  auto vfs = shaders_vfs;
  vfs.WriteFile("material_sampler.hlsli", CodeLines{shaders_vfs, "material/lambertian/sampler.hlsl"});
  vfs.WriteFile("entity_chit.hlsl", CodeLines{shaders_vfs, "geometry/mesh/hit_group.hlsl"});
  core_.GraphicsCore()->CreateShader(vfs, "entity_chit.hlsl", "RenderClosestHit", "lib_6_5", {"-I."}, &shader);
  shaders_["mesh_lambertian_chit"] = std::move(shader);
  core_.GraphicsCore()->CreateShader(vfs, "entity_chit.hlsl", "ShadowClosestHit", "lib_6_5", {"-I."}, &shader);
  shaders_["mesh_lambertian_shadow_chit"] = std::move(shader);

  vfs.WriteFile("material_sampler.hlsli", CodeLines{shaders_vfs, "material/light/sampler.hlsl"});
  core_.GraphicsCore()->CreateShader(vfs, "entity_chit.hlsl", "RenderClosestHit", "lib_6_5", {"-I."}, &shader);
  shaders_["mesh_light_chit"] = std::move(shader);
  core_.GraphicsCore()->CreateShader(vfs, "entity_chit.hlsl", "ShadowClosestHit", "lib_6_5", {"-I."}, &shader);
  shaders_["mesh_light_shadow_chit"] = std::move(shader);
  core_.GraphicsCore()->CreateShader(vfs, "entity_chit.hlsl", "ShadowAnyHit", "lib_6_5", {"-I."}, &shader);
  shaders_["mesh_light_shadow_ahit"] = std::move(shader);

  vfs.WriteFile("material_sampler.hlsli", CodeLines{shaders_vfs, "material/principled/sampler.hlsl"});
  core_.GraphicsCore()->CreateShader(vfs, "entity_chit.hlsl", "RenderClosestHit", "lib_6_5", {"-I."}, &shader);
  shaders_["mesh_principled_chit"] = std::move(shader);
  core_.GraphicsCore()->CreateShader(vfs, "entity_chit.hlsl", "ShadowClosestHit", "lib_6_5", {"-I."}, &shader);
  shaders_["mesh_principled_shadow_chit"] = std::move(shader);

  vfs.WriteFile("material_sampler.hlsli", CodeLines{shaders_vfs, "material/specular/sampler.hlsl"});
  core_.GraphicsCore()->CreateShader(vfs, "entity_chit.hlsl", "RenderClosestHit", "lib_6_5", {"-I."}, &shader);
  shaders_["mesh_specular_chit"] = std::move(shader);
  core_.GraphicsCore()->CreateShader(vfs, "entity_chit.hlsl", "ShadowClosestHit", "lib_6_5", {"-I."}, &shader);
  shaders_["mesh_specular_shadow_chit"] = std::move(shader);
}

Core *DedicatedCast(sparkium::Core *core) {
  COMPONENT_CAST(core, Core)
  return nullptr;
}

void Render(sparkium::Core *core,
            sparkium::Scene *scene,
            sparkium::Camera *camera,
            sparkium::Film *film,
            bool software,
            bool ray_query) {
  DedicatedCast(core);
  DedicatedCast(scene)->Render(DedicatedCast(camera), DedicatedCast(film), software, ray_query);
}
}  // namespace sparkium::raytracing
