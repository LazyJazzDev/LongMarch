#include "sparkium/pipelines/raytracing/core/core.h"

#include "sparkium/pipelines/raytracing/core/camera.h"
#include "sparkium/pipelines/raytracing/core/film.h"
#include "sparkium/pipelines/raytracing/core/scene.h"

namespace sparkium::raytracing {

Core::Core(sparkium::Core &core) : core_(core) {
  LoadPublicShaders();
}

rhi::IDevice *Core::GetDevice() const {
  return core_.GetDevice();
}

const VirtualFileSystem &Core::GetShadersVFS() const {
  return core_.GetShadersVFS();
}

rhi::IShader *Core::GetShader(const std::string &name) {
  return core_.GetShader(name);
}

rhi::IComputeProgram *Core::GetComputeProgram(const std::string &name) {
  return core_.GetComputeProgram(name);
}

rhi::IComputeProgram *Core::GetGeometryLightPowerProgram(const CodeLines &geometry, const CodeLines &material) {
  // The VFS, backend, entry point, profile and binding layout are fixed for this Core.
  auto key = std::make_pair(std::string(geometry), std::string(material));
  auto found = geometry_light_programs_.find(key);
  if (found != geometry_light_programs_.end())
    return found->second.program.Get();
  GeometryLightResources resources;
  auto vfs = GetShadersVFS();
  vfs.WriteFile("geometry_sampler.slang", geometry);
  vfs.WriteFile("material_evaluator.slang", material);
  resources.shader =
      GetDevice()->CreateShader("light/geometry_material/gather_primitive_power.slang", "GatherPrimitivePowerKernel",
                                rhi::SHADER_STAGE_COMPUTE, &vfs, {"."}, {});
  if (!resources.shader)
    throw std::runtime_error("failed to compile geometry light power shader");
  {
    rhi::ComputeProgramDesc desc;
    desc.compute_shader = resources.shader.Get();
    desc.pipeline_layout.groups = {{{{0, rhi::BINDING_BYTE_ADDRESS_BUFFER, 1},
                                     {1, rhi::BINDING_BYTE_ADDRESS_BUFFER, 1},
                                     {2, rhi::BINDING_RW_BYTE_ADDRESS_BUFFER, 1}}}};
    resources.program = GetDevice()->CreateComputeProgram(desc);
  }
  auto *program = resources.program.Get();
  geometry_light_programs_.emplace(std::move(key), std::move(resources));
  return program;
}

rhi::IImage *Core::GetImage(const std::string &name) {
  return core_.GetImage(name);
}

rhi::IBuffer *Core::GetBuffer(const std::string &name) {
  return core_.GetBuffer(name);
}

void Core::LoadPublicShaders() {
  rhi::ObjPtr<rhi::IShader> shader;
  rhi::ObjPtr<rhi::IComputeProgram> compute_program;
  auto &shaders_vfs = core_.GetShadersVFS();
  shader =
      core_.GetDevice()->CreateShader("film2img.slang", "Main", rhi::SHADER_STAGE_COMPUTE, &shaders_vfs, {"."}, {});
  core_.SetPublicResource("film2img", std::move(shader));

  {
    rhi::ComputeProgramDesc desc;
    desc.compute_shader = core_.GetShader("film2img");
    desc.pipeline_layout.groups = {
        {{{0, rhi::BINDING_TEXTURE2D, 1}, {1, rhi::BINDING_TEXTURE2D, 1}, {2, rhi::BINDING_RW_TEXTURE2D, 1}}}};
    compute_program = core_.GetDevice()->CreateComputeProgram(desc);
  }
  core_.SetPublicResource("film2img", std::move(compute_program));

  shader = core_.GetDevice()->CreateShader("blelloch_scan.slang", "BlellochUpSweep", rhi::SHADER_STAGE_COMPUTE,
                                           &shaders_vfs, {"."}, {});
  core_.SetPublicResource("blelloch_scan_up", std::move(shader));
  shader = core_.GetDevice()->CreateShader("blelloch_scan.slang", "BlellochDownSweep", rhi::SHADER_STAGE_COMPUTE,
                                           &shaders_vfs, {"."}, {});
  core_.SetPublicResource("blelloch_scan_down", std::move(shader));

  {
    rhi::ComputeProgramDesc desc;
    desc.compute_shader = core_.GetShader("blelloch_scan_up");
    desc.pipeline_layout.groups = {
        {{{0, rhi::BINDING_RW_BYTE_ADDRESS_BUFFER, 1}, {1, rhi::BINDING_UNIFORM_BUFFER, 1}}}};
    compute_program = core_.GetDevice()->CreateComputeProgram(desc);
  }
  core_.SetPublicResource("blelloch_scan_up", std::move(compute_program));

  {
    rhi::ComputeProgramDesc desc;
    desc.compute_shader = core_.GetShader("blelloch_scan_down");
    desc.pipeline_layout.groups = {
        {{{0, rhi::BINDING_RW_BYTE_ADDRESS_BUFFER, 1}, {1, rhi::BINDING_UNIFORM_BUFFER, 1}}}};
    compute_program = core_.GetDevice()->CreateComputeProgram(desc);
  }
  core_.SetPublicResource("blelloch_scan_down", std::move(compute_program));
}

void Core::PrepareBuiltinHitShaders() {
  if (native_shaders_ready_)
    return;
  rhi::ObjPtr<rhi::IShader> shader;
  auto &shaders_vfs = core_.GetShadersVFS();
  if (!core_.GetDevice()->SupportRayTracingPipeline())
    return;

  auto vfs = shaders_vfs;
  vfs.WriteFile("material_sampler.slang", CodeLines{shaders_vfs, "material/lambertian/sampler.slang"});
  vfs.WriteFile("entity_chit.slang", CodeLines{shaders_vfs, "geometry/mesh/hit_group.slang"});
  shader = core_.GetDevice()->CreateShader("entity_chit.slang", "RenderClosestHit", rhi::SHADER_STAGE_CLOSEST_HIT, &vfs,
                                           {"."}, {});
  core_.SetPublicResource("mesh_lambertian_chit", std::move(shader));
  shader = core_.GetDevice()->CreateShader("entity_chit.slang", "ShadowClosestHit", rhi::SHADER_STAGE_CLOSEST_HIT, &vfs,
                                           {"."}, {});
  core_.SetPublicResource("mesh_lambertian_shadow_chit", std::move(shader));

  vfs.WriteFile("material_sampler.slang", CodeLines{shaders_vfs, "material/light/sampler.slang"});
  shader = core_.GetDevice()->CreateShader("entity_chit.slang", "RenderClosestHit", rhi::SHADER_STAGE_CLOSEST_HIT, &vfs,
                                           {"."}, {});
  core_.SetPublicResource("mesh_light_chit", std::move(shader));
  shader = core_.GetDevice()->CreateShader("entity_chit.slang", "ShadowClosestHit", rhi::SHADER_STAGE_CLOSEST_HIT, &vfs,
                                           {"."}, {});
  core_.SetPublicResource("mesh_light_shadow_chit", std::move(shader));
  shader =
      core_.GetDevice()->CreateShader("entity_chit.slang", "ShadowAnyHit", rhi::SHADER_STAGE_ANY_HIT, &vfs, {"."}, {});
  core_.SetPublicResource("mesh_light_shadow_ahit", std::move(shader));

  vfs.WriteFile("material_sampler.slang", CodeLines{shaders_vfs, "material/principled/sampler.slang"});
  shader = core_.GetDevice()->CreateShader("entity_chit.slang", "RenderClosestHit", rhi::SHADER_STAGE_CLOSEST_HIT, &vfs,
                                           {"."}, {});
  core_.SetPublicResource("mesh_principled_chit", std::move(shader));
  shader = core_.GetDevice()->CreateShader("entity_chit.slang", "ShadowClosestHit", rhi::SHADER_STAGE_CLOSEST_HIT, &vfs,
                                           {"."}, {});
  core_.SetPublicResource("mesh_principled_shadow_chit", std::move(shader));

  vfs.WriteFile("material_sampler.slang", CodeLines{shaders_vfs, "material/specular/sampler.slang"});
  shader = core_.GetDevice()->CreateShader("entity_chit.slang", "RenderClosestHit", rhi::SHADER_STAGE_CLOSEST_HIT, &vfs,
                                           {"."}, {});
  core_.SetPublicResource("mesh_specular_chit", std::move(shader));
  shader = core_.GetDevice()->CreateShader("entity_chit.slang", "ShadowClosestHit", rhi::SHADER_STAGE_CLOSEST_HIT, &vfs,
                                           {"."}, {});
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
