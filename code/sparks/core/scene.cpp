#include "sparks/core/scene.h"

#include "sparks/core/camera.h"
#include "sparks/core/core.h"
#include "sparks/core/entity.h"
#include "sparks/core/film.h"
#include "sparks/core/geometry.h"
#include "sparks/core/surface.h"

namespace sparks {
Scene::Scene(Core *core) : core_(core) {
  core_->GraphicsCore()->CreateShader(core_->GetShadersVFS(), "raygen.hlsl", "Main", "lib_6_3", &raygen_shader_);
  core_->GraphicsCore()->CreateShader(core_->GetShadersVFS(), "raygen.hlsl", "MissMain", "lib_6_3", &miss_shader_);
  core_->GraphicsCore()->CreateBuffer(sizeof(Settings), graphics::BUFFER_TYPE_STATIC, &scene_settings_buffer_);
}

void Scene::Render(Camera *camera, Film *film) {
  UpdatePipeline(camera);
  scene_settings_buffer_->UploadData(&settings, sizeof(Settings));
  std::unique_ptr<graphics::CommandContext> cmd_context;
  core_->GraphicsCore()->CreateCommandContext(&cmd_context);
  cmd_context->CmdBindRayTracingProgram(rt_program_.get());
  cmd_context->CmdBindResources(0, {film->accumulated_color_.get()}, graphics::BIND_POINT_RAYTRACING);
  cmd_context->CmdBindResources(1, {film->accumulated_samples_.get()}, graphics::BIND_POINT_RAYTRACING);
  cmd_context->CmdBindResources(2, {tlas_.get()}, graphics::BIND_POINT_RAYTRACING);
  cmd_context->CmdBindResources(3, {scene_settings_buffer_.get()}, graphics::BIND_POINT_RAYTRACING);
  cmd_context->CmdBindResources(4, {camera->Buffer()}, graphics::BIND_POINT_RAYTRACING);
  cmd_context->CmdBindResources(5, buffers_, graphics::BIND_POINT_RAYTRACING);
  cmd_context->CmdBindResources(6, {entity_info_buffer_.get()}, graphics::BIND_POINT_RAYTRACING);
  cmd_context->CmdDispatchRays(film->accumulated_color_->Extent().width, film->accumulated_samples_->Extent().height,
                               1);
  core_->GraphicsCore()->SubmitCommandContext(cmd_context.get());
  core_->GraphicsCore()->WaitGPU();
}

void Scene::AddEntity(Entity *entity) {
  entities_.insert(entity);
}

GeometryRegistration Scene::RegisterGeometry(Geometry *geometry) {
  GeometryRegistration geom_reg{};
  auto data = geometry->Buffer();
  auto blas = geometry->BLAS();
  auto hit_group = geometry->HitGroup();
  geom_reg.data_index = RegisterBuffer(data);

  geom_reg.blas = blas;

  if (!hit_group_map_.count(hit_group)) {
    hit_group_map_[hit_group] = static_cast<int32_t>(hit_groups_.size());
    hit_groups_.emplace_back(hit_group);
  }
  geom_reg.hit_group_index = hit_group_map_[hit_group];

  return geom_reg;
}

SurfaceRegistration Scene::RegisterSurface(Surface *surface) {
  SurfaceRegistration surf_reg{};
  surf_reg.shader_index = RegisterCallableShader(surface->CallableShader());
  surf_reg.data_index = RegisterBuffer(surface->Buffer());
  return surf_reg;
}

InstanceRegistration Scene::RegisterInstance(GeometryRegistration geom_reg,
                                             const glm::mat4x3 &transformation,
                                             SurfaceRegistration surf_reg) {
  InstanceRegistration instance_reg;
  instance_reg.instance_index = instances_.size();

  instances_.push_back(geom_reg.blas->MakeInstance(transformation, geom_reg.data_index, 255, geom_reg.hit_group_index));
  surfaces_registrations_.push_back(surf_reg);

  return instance_reg;
}

int32_t Scene::RegisterCallableShader(graphics::Shader *callable_shader) {
  if (!callable_shader_map_.count(callable_shader)) {
    callable_shader_map_[callable_shader] = static_cast<int32_t>(callable_shaders_.size());
    callable_shaders_.emplace_back(callable_shader);
  }
  return callable_shader_map_[callable_shader];
}

int32_t Scene::RegisterBuffer(graphics::Buffer *buffer) {
  if (!buffer_map_.count(buffer)) {
    buffer_map_[buffer] = static_cast<int32_t>(buffers_.size());
    buffers_.emplace_back(buffer);
  }
  return buffer_map_[buffer];
}

void Scene::UpdatePipeline(Camera *camera) {
  rt_program_.reset();
  tlas_.reset();
  core_->GraphicsCore()->CreateRayTracingProgram(&rt_program_);
  miss_shader_indices_ = {0};
  hit_groups_.clear();
  hit_group_map_.clear();
  buffers_.clear();
  buffer_map_.clear();
  callable_shaders_.clear();
  callable_shader_map_.clear();
  instances_.clear();
  surfaces_registrations_.clear();

  RegisterCallableShader(camera->Shader());

  for (auto entity : entities_) {
    entity->Update(this);
  }
  callable_shader_indices_.resize(callable_shaders_.size());
  std::iota(callable_shader_indices_.begin(), callable_shader_indices_.end(), 0);
  hit_group_indices_.resize(hit_groups_.size());
  std::iota(hit_group_indices_.begin(), hit_group_indices_.end(), 0);

  core_->GraphicsCore()->CreateTopLevelAccelerationStructure(instances_, &tlas_);

  entity_info_buffer_.reset();
  core_->GraphicsCore()->CreateBuffer(sizeof(SurfaceRegistration) * surfaces_registrations_.size(),
                                      graphics::BUFFER_TYPE_STATIC, &entity_info_buffer_);
  entity_info_buffer_->UploadData(surfaces_registrations_.data(),
                                  sizeof(SurfaceRegistration) * surfaces_registrations_.size());

  rt_program_->AddResourceBinding(graphics::RESOURCE_TYPE_IMAGE, 1);
  rt_program_->AddResourceBinding(graphics::RESOURCE_TYPE_IMAGE, 1);
  rt_program_->AddResourceBinding(graphics::RESOURCE_TYPE_ACCELERATION_STRUCTURE, 1);
  rt_program_->AddResourceBinding(graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);
  rt_program_->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, 1);
  rt_program_->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, buffers_.size());
  rt_program_->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, 1);

  rt_program_->AddRayGenShader(raygen_shader_.get());
  rt_program_->AddMissShader(miss_shader_.get());
  for (const auto &hit_group : hit_groups_) {
    rt_program_->AddHitGroup(hit_group);
  }
  for (const auto &callable_shader : callable_shaders_) {
    rt_program_->AddCallableShader(callable_shader);
  }
  rt_program_->Finalize(miss_shader_indices_, hit_group_indices_, callable_shader_indices_);
}

}  // namespace sparks
