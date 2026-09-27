#include "sparkium/pipelines/raytracing/core/core.h"

#include "sparkium/pipelines/raytracing/core/camera.h"
#include "sparkium/pipelines/raytracing/core/film.h"
#include "sparkium/pipelines/raytracing/core/scene.h"

namespace sparkium::raytracing {

Core::Core(sparkium::Core &core) : core_(core) {
  LoadPublicShaders();
}

graphics::Core *Core::GraphicsCore() const {
  return core_.GraphicsCore();
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

graphics::ComputeProgram *Core::GeometryLightProgram(const CodeLines &geometry, const CodeLines &material) {
  // The VFS, backend, entry point, profile and binding layout are fixed for this Core.
  auto key = std::make_pair(std::string(geometry), std::string(material));
  auto found = geometry_light_programs_.find(key);
  if (found != geometry_light_programs_.end())
    return found->second.program.get();
  GeometryLightResources resources;
  auto vfs = GetShadersVFS();
  vfs.WriteFile("geometry_sampler.slang", geometry);
  vfs.WriteFile("material_evaluator.slang", material);
  if (GraphicsCore()->CreateShader(vfs, "light/geometry_material/gather_primitive_power.slang",
                                   "GatherPrimitivePowerKernel", "cs_6_3", {"-I."}, &resources.shader) != 0)
    throw std::runtime_error("failed to compile geometry light power shader");
  GraphicsCore()->CreateComputeProgram(resources.shader.get(), &resources.program);
  resources.program->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, 1);
  resources.program->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, 1);
  resources.program->AddResourceBinding(graphics::RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER, 1);
  resources.program->Finalize();
  auto *program = resources.program.get();
  geometry_light_programs_.emplace(std::move(key), std::move(resources));
  return program;
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
  core_.GraphicsCore()->CreateShader(shaders_vfs, "film2img.slang", "Main", "cs_6_0", &shader);
  core_.SetPublicResource("film2img", std::move(shader));

  core_.GraphicsCore()->CreateComputeProgram(core_.GetShader("film2img"), &compute_program);
  compute_program->AddResourceBinding(graphics::RESOURCE_TYPE_IMAGE, 1);
  compute_program->AddResourceBinding(graphics::RESOURCE_TYPE_IMAGE, 1);
  compute_program->AddResourceBinding(graphics::RESOURCE_TYPE_WRITABLE_IMAGE, 1);
  compute_program->Finalize();
  core_.SetPublicResource("film2img", std::move(compute_program));

  core_.GraphicsCore()->CreateShader(shaders_vfs, "blelloch_scan.slang", "BlellochUpSweep", "cs_6_3", {"-I."}, &shader);
  core_.SetPublicResource("blelloch_scan_up", std::move(shader));
  core_.GraphicsCore()->CreateShader(shaders_vfs, "blelloch_scan.slang", "BlellochDownSweep", "cs_6_3", {"-I."},
                                     &shader);
  core_.SetPublicResource("blelloch_scan_down", std::move(shader));

  core_.GraphicsCore()->CreateComputeProgram(core_.GetShader("blelloch_scan_up"), &compute_program);
  compute_program->AddResourceBinding(graphics::RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER, 1);
  compute_program->AddResourceBinding(graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);
  compute_program->Finalize();
  core_.SetPublicResource("blelloch_scan_up", std::move(compute_program));

  core_.GraphicsCore()->CreateComputeProgram(core_.GetShader("blelloch_scan_down"), &compute_program);
  compute_program->AddResourceBinding(graphics::RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER, 1);
  compute_program->AddResourceBinding(graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);
  compute_program->Finalize();
  core_.SetPublicResource("blelloch_scan_down", std::move(compute_program));
}

void Core::PrepareNativeShaders() {
  if (native_shaders_ready_)
    return;
  std::unique_ptr<graphics::Shader> shader;
  auto &shaders_vfs = core_.GetShadersVFS();
  if (!core_.GraphicsCore()->DeviceRayTracingSupport())
    return;

  auto vfs = shaders_vfs;
  vfs.WriteFile("material_sampler.slang", CodeLines{shaders_vfs, "material/lambertian/sampler.slang"});
  vfs.WriteFile("entity_chit.slang", CodeLines{shaders_vfs, "geometry/mesh/hit_group.slang"});
  core_.GraphicsCore()->CreateShader(vfs, "entity_chit.slang", "RenderClosestHit", "lib_6_5", {"-I."}, &shader);
  core_.SetPublicResource("mesh_lambertian_chit", std::move(shader));
  core_.GraphicsCore()->CreateShader(vfs, "entity_chit.slang", "ShadowClosestHit", "lib_6_5", {"-I."}, &shader);
  core_.SetPublicResource("mesh_lambertian_shadow_chit", std::move(shader));

  vfs.WriteFile("material_sampler.slang", CodeLines{shaders_vfs, "material/light/sampler.slang"});
  core_.GraphicsCore()->CreateShader(vfs, "entity_chit.slang", "RenderClosestHit", "lib_6_5", {"-I."}, &shader);
  core_.SetPublicResource("mesh_light_chit", std::move(shader));
  core_.GraphicsCore()->CreateShader(vfs, "entity_chit.slang", "ShadowClosestHit", "lib_6_5", {"-I."}, &shader);
  core_.SetPublicResource("mesh_light_shadow_chit", std::move(shader));
  core_.GraphicsCore()->CreateShader(vfs, "entity_chit.slang", "ShadowAnyHit", "lib_6_5", {"-I."}, &shader);
  core_.SetPublicResource("mesh_light_shadow_ahit", std::move(shader));

  vfs.WriteFile("material_sampler.slang", CodeLines{shaders_vfs, "material/principled/sampler.slang"});
  core_.GraphicsCore()->CreateShader(vfs, "entity_chit.slang", "RenderClosestHit", "lib_6_5", {"-I."}, &shader);
  core_.SetPublicResource("mesh_principled_chit", std::move(shader));
  core_.GraphicsCore()->CreateShader(vfs, "entity_chit.slang", "ShadowClosestHit", "lib_6_5", {"-I."}, &shader);
  core_.SetPublicResource("mesh_principled_shadow_chit", std::move(shader));

  vfs.WriteFile("material_sampler.slang", CodeLines{shaders_vfs, "material/specular/sampler.slang"});
  core_.GraphicsCore()->CreateShader(vfs, "entity_chit.slang", "RenderClosestHit", "lib_6_5", {"-I."}, &shader);
  core_.SetPublicResource("mesh_specular_chit", std::move(shader));
  core_.GraphicsCore()->CreateShader(vfs, "entity_chit.slang", "ShadowClosestHit", "lib_6_5", {"-I."}, &shader);
  core_.SetPublicResource("mesh_specular_shadow_chit", std::move(shader));
  native_shaders_ready_ = true;
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
  auto rt_core = DedicatedCast(core);
  auto rt_scene = DedicatedCast(scene);
  auto rt_film = DedicatedCast(film);
  auto rt_camera = DedicatedCast(camera);
  rt_scene->Render(rt_camera, rt_film, software, ray_query);
}

}  // namespace sparkium::raytracing
