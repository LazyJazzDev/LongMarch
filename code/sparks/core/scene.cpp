#include "sparks/core/scene.h"

#include "film.h"
#include "sparks/core/camera.h"
#include "sparks/core/core.h"
#include "sparks/core/entity.h"
#include "sparks/core/film.h"
#include "sparks/core/geometry.h"
#include "sparks/core/material.h"

namespace sparks {
Scene::Scene(Core *core) : core_(core) {
  core_->GraphicsCore()->CreateShader(core_->GetShadersVFS(), "raygen.hlsl", "Main", "lib_6_5", &raygen_shader_);
  core_->GraphicsCore()->CreateShader(core_->GetShadersVFS(), "raygen.hlsl", "MissMain", "lib_6_5",
                                      &default_miss_shader_);
  core_->GraphicsCore()->CreateBuffer(sizeof(Settings) + sizeof(Film::Info), graphics::BUFFER_TYPE_STATIC,
                                      &scene_settings_buffer_);
  core_->GraphicsCore()->CreateShader(core_->GetShadersVFS(), "gather_light_power.hlsl", "GatherLightPowerKernel",
                                      "cs_6_3", &gather_light_power_shader_);
}

void Scene::Render(Camera *camera, Film *film) {
  UpdatePipeline(camera);
  scene_settings_buffer_->UploadData(&settings, sizeof(Settings));
  scene_settings_buffer_->UploadData(&film->info, sizeof(Film::Info), sizeof(Settings));
  film->info.accumulated_samples += settings.samples_per_dispatch;
  std::unique_ptr<graphics::CommandContext> cmd_context;
  core_->GraphicsCore()->CreateCommandContext(&cmd_context);
  cmd_context->CmdBindRayTracingProgram(rt_program_.get());
  cmd_context->CmdBindResources(0, {film->accumulated_color_.get()}, graphics::BIND_POINT_RAYTRACING);
  cmd_context->CmdBindResources(1, {film->accumulated_samples_.get()}, graphics::BIND_POINT_RAYTRACING);
  cmd_context->CmdBindResources(2, {tlas_.get()}, graphics::BIND_POINT_RAYTRACING);
  cmd_context->CmdBindResources(3, {scene_settings_buffer_.get()}, graphics::BIND_POINT_RAYTRACING);
  cmd_context->CmdBindResources(4, {core_->SobolBuffer()}, graphics::BIND_POINT_RAYTRACING);
  cmd_context->CmdBindResources(5, {camera->Buffer()}, graphics::BIND_POINT_RAYTRACING);
  cmd_context->CmdBindResources(6, buffers_, graphics::BIND_POINT_RAYTRACING);
  cmd_context->CmdBindResources(7, {instance_metadata_buffer_.get()}, graphics::BIND_POINT_RAYTRACING);
  cmd_context->CmdBindResources(8, {light_selector_buffer_.get()}, graphics::BIND_POINT_RAYTRACING);
  cmd_context->CmdBindResources(9, {light_metadatas_buffer_.get()}, graphics::BIND_POINT_RAYTRACING);
  cmd_context->CmdDispatchRays(film->accumulated_color_->Extent().width, film->accumulated_samples_->Extent().height,
                               1);
  core_->GraphicsCore()->SubmitCommandContext(cmd_context.get());
  core_->GraphicsCore()->WaitGPU();
}

void Scene::AddEntity(Entity *entity) {
  entities_.insert({entity, true});
}

void Scene::DeleteEntity(Entity *entity) {
  entities_.erase(entity);
  buffers_dirty_ = true;
  hit_groups_dirty_ = true;
  callable_shaders_dirty_ = true;
}

void Scene::SetEntityActive(Entity *entity, bool active) {
  entities_.at(entity) = active;
}

int32_t Scene::RegisterInstance(graphics::AccelerationStructure *blas,
                                const glm::mat4x3 &transformation,
                                int32_t hit_group_index,
                                int32_t geometry_data_index,
                                int32_t material_data_index,
                                int32_t custom_index) {
  int32_t instance_reg = instances_.size();
  InstanceMetadata entity_metadata{};
  instances_.push_back(blas->MakeInstance(transformation, geometry_data_index, 255, hit_group_index));
  entity_metadata.geometry_data_index = geometry_data_index;
  entity_metadata.material_data_index = material_data_index;
  entity_metadata.custom_index = custom_index;
  instance_metadatas_.push_back(entity_metadata);
  return instance_reg;
}

int32_t Scene::RegisterInstance(const GeometryRegistration &geom_reg,
                                const glm::mat4x3 &transformation,
                                const MaterialRegistration &mat_reg,
                                int32_t custom_index) {
  int instance_reg = instances_.size();
  InstanceMetadata metadata{};
  metadata.material_data_index = mat_reg.data_index;
  metadata.material_shader_index = mat_reg.shader_index;
  metadata.geometry_data_index = geom_reg.data_index;
  metadata.geometry_shader_index = geom_reg.shader_index;
  metadata.custom_index = custom_index;
  instances_.push_back(geom_reg.blas->MakeInstance(transformation, geom_reg.data_index, 255, geom_reg.hit_group_index));
  instance_metadatas_.push_back(metadata);
  return instance_reg;
}

int &Scene::LightCustomIndex(int32_t light_index) {
  return light_metadatas_[light_index].custom_index;
}

int &Scene::InstanceCustomIndex(int32_t instance_index) {
  return instance_metadatas_[instance_index].custom_index;
}

int32_t Scene::RegisterCallableShader(graphics::Shader *callable_shader) {
  if (!callable_shader)
    return -1;
  if (!callable_shader_map_.count(callable_shader)) {
    callable_shader_map_[callable_shader] = static_cast<int32_t>(callable_shaders_.size());
    callable_shaders_.emplace_back(callable_shader);
    callable_shaders_dirty_ = true;
  }
  return callable_shader_map_[callable_shader];
}

int32_t Scene::RegisterBuffer(graphics::Buffer *buffer) {
  if (!buffer)
    return -1;
  if (!buffer_map_.count(buffer)) {
    buffer_map_[buffer] = static_cast<int32_t>(buffers_.size());
    buffers_.emplace_back(buffer);
    buffers_dirty_ = true;
  }
  return buffer_map_[buffer];
}

int32_t Scene::RegisterHitGroup(const graphics::HitGroup &hit_group) {
  if (!hit_group_map_.count(hit_group)) {
    hit_group_map_[hit_group] = static_cast<int32_t>(hit_groups_.size());
    hit_groups_.emplace_back(hit_group);
    hit_groups_dirty_ = true;
  }
  return hit_group_map_[hit_group];
}

GeometryRegistration Scene::RegisterGeometry(Geometry *geometry) {
  GeometryRegistration reg;
  reg.data_index = RegisterBuffer(geometry->Buffer());
  auto sampler_impl = geometry->SamplerImpl();
  auto hit_record_impl = geometry->HitRecordImpl();
  std::string geometry_code = sampler_impl;
  geometry_code += std::string(hit_record_impl);
  if (!geometry_shader_map_.count(geometry_code)) {
    sampler_impl.InsertFront("#define PrimitiveArea PrimitiveArea" + std::to_string(geometry_shader_index_));
    sampler_impl.InsertFront("#define SamplePrimitive SamplePrimitive" + std::to_string(geometry_shader_index_));

    sampler_impl.InsertBack("#undef PrimitiveArea");
    sampler_impl.InsertBack("#undef SamplePrimitive");
    geometry_shader_assembled_.InsertAfter(sampler_impl, "// Geometry Sampler Implementation");
    geometry_shader_assembled_.InsertAfter("    return PrimitiveArea" + std::to_string(geometry_shader_index_) +
                                               "(geometry_data, transform, primitive_id);",
                                           "// PrimitiveArea Function List");
    geometry_shader_assembled_.InsertAfter("  case " + std::to_string(geometry_shader_index_) + ":",
                                           "// PrimitiveArea Function List");
    geometry_shader_assembled_.InsertAfter("    return SamplePrimitive" + std::to_string(geometry_shader_index_) +
                                               "(geometry_data, transform, primitive_id, sample);",
                                           "// SamplePrimitive Function List");
    geometry_shader_assembled_.InsertAfter("  case " + std::to_string(geometry_shader_index_) + ":",
                                           "// SamplePrimitive Function List");

    hit_record_impl.InsertFront("#define GetHitRecord GetHitRecord" + std::to_string(geometry_shader_index_));
    hit_record_impl.InsertBack("#undef GetHitRecord");

    hit_record_assembled_.InsertAfter(hit_record_impl, "// Hit Record Implementation");
    hit_record_assembled_.InsertAfter(
        "    return GetHitRecord" + std::to_string(geometry_shader_index_) + "(payload, direction);",
        "// GetHitRecord Function List");
    hit_record_assembled_.InsertAfter("  case " + std::to_string(geometry_shader_index_) + ":",
                                      "// GetHitRecord Function List");

    geometry_shader_map_[geometry_code] = geometry_shader_index_++;
  }
  reg.shader_index = geometry_shader_map_[geometry_code];
  reg.hit_group_index = RegisterHitGroup(geometry->HitGroup());
  reg.blas = geometry->BLAS();
  return reg;
}

MaterialRegistration Scene::RegisterMaterial(Material *material) {
  MaterialRegistration reg;
  reg.data_index = RegisterBuffer(material->Buffer());
  auto material_sampler_impl = material->SamplerImpl();
  auto material_direct_lighting_evaluate_impl = material->EvaluatorImpl();
  std::string material_code = std::string(material_sampler_impl) + std::string(material_direct_lighting_evaluate_impl);
  if (!material_shader_map_.count(material_code)) {
    material_sampler_impl.InsertFront("#define SampleMaterial SampleMaterial" + std::to_string(material_shader_index_));
    material_sampler_impl.InsertBack("#undef SampleMaterial");

    material_shader_assembled_.InsertAfter(material_sampler_impl, "// Material Sampler Implementation");
    material_shader_assembled_.InsertAfter(
        "    return SampleMaterial" + std::to_string(material_shader_index_) + "(context, hit_record);",
        "// SampleMaterial Function List");
    material_shader_assembled_.InsertAfter("  case " + std::to_string(material_shader_index_) + ":",
                                           "// SampleMaterial Function List");

    material_direct_lighting_evaluate_impl.InsertFront("#define EvaluateDirectLighting EvaluateDirectLighting" +
                                                       std::to_string(material_shader_index_));
    material_direct_lighting_evaluate_impl.InsertBack("#undef EvaluateDirectLighting");

    material_shader_assembled_.InsertAfter(material_direct_lighting_evaluate_impl,
                                           "// Evaluate Direct Lighting Implementation");

    material_shader_assembled_.InsertAfter("    return EvaluateDirectLighting" +
                                               std::to_string(material_shader_index_) +
                                               "(material_data, position, primitive_sample);",
                                           "// EvaluateDirectLighting Function List");
    material_shader_assembled_.InsertAfter("  case " + std::to_string(material_shader_index_) + ":",
                                           "// EvaluateDirectLighting Function List");

    material_shader_map_[material_code] = material_shader_index_++;
  }
  reg.shader_index = material_shader_map_[material_code];
  return reg;
}

int32_t Scene::RegisterLight(Light *light, int custom_index) {
  int32_t light_reg_index = light_metadatas_.size();
  LightMetadata light_reg{};
  light_reg.sampler_data_index = RegisterBuffer(light->SamplerData());
  light_reg.sampler_shader_index = RegisterCallableShader(light->SamplerShader());
  light_reg.custom_index = custom_index;
  light_reg.power_offset = light->SamplerPreprocess(preprocess_cmd_context_.get());
  light_metadatas_.push_back(light_reg);

  CodeLines light_sampler_impl = light->SamplerImpl();
  std::string light_code = std::string(light_sampler_impl);
  if (!light_shader_map_.count(light_code)) {
    light_sampler_impl.InsertFront("#define LightSampler LightSampler" + std::to_string(light_shader_index_));
    light_sampler_impl.InsertBack("#undef LightSampler");

    light_shader_assembled_.InsertAfter(light_sampler_impl, "// Light Sampler Implementation");
    light_shader_assembled_.InsertAfter("    return LightSampler" + std::to_string(light_shader_index_) + "(payload);",
                                        "// LightSampler Function List");
    light_shader_assembled_.InsertAfter("  case " + std::to_string(light_shader_index_) + ":",
                                        "// LightSampler Function List");

    light_shader_map_[light_code] = light_shader_index_++;
  }

  return light_reg_index;
}

void Scene::UpdatePipeline(Camera *camera) {
  instances_.clear();
  instance_metadatas_.clear();
  light_metadatas_.clear();

  preprocess_cmd_context_.reset();
  core_->GraphicsCore()->CreateCommandContext(&preprocess_cmd_context_);

  bool &clean_buffers = buffers_dirty_;
  bool &clean_hit_groups = hit_groups_dirty_;
  bool &clean_callable_shaders = callable_shaders_dirty_;

  for (auto [entity, active] : entities_) {
    clean_buffers |= entity->ExpiredBuffer();
    clean_hit_groups |= entity->ExpiredHitGroup();
    clean_callable_shaders |= entity->ExpiredCallableShader();
  }

  if (clean_buffers) {
    buffers_.clear();
    buffer_map_.clear();
  }

  if (clean_hit_groups) {
    hit_groups_.clear();
    hit_group_map_.clear();
  }

  if (clean_callable_shaders) {
    callable_shaders_.clear();
    callable_shader_map_.clear();
    RegisterCallableShader(camera->Shader());
  }

  geometry_shader_assembled_ = {core_->GetShadersVFS(), "geometry_shaders.hlsli"};
  hit_record_assembled_ = {core_->GetShadersVFS(), "hit_record.hlsli"};
  geometry_shader_map_.clear();
  geometry_shader_index_ = 0;

  material_shader_assembled_ = {core_->GetShadersVFS(), "material_shaders.hlsli"};
  material_shader_map_.clear();
  material_shader_index_ = 0;

  light_shader_assembled_ = {core_->GetShadersVFS(), "light_shaders.hlsli"};
  light_shader_map_.clear();
  light_shader_index_ = 0;

  for (auto [entity, active] : entities_) {
    if (active) {
      entity->Update(this);
    }
  }

  if (!tlas_) {
    core_->GraphicsCore()->CreateTopLevelAccelerationStructure(instances_, &tlas_);
  } else {
    tlas_->UpdateInstances(instances_);
  }

  if (!instance_metadata_buffer_ ||
      sizeof(InstanceMetadata) * instance_metadatas_.size() > instance_metadata_buffer_->Size()) {
    instance_metadata_buffer_.reset();
    core_->GraphicsCore()->CreateBuffer(sizeof(InstanceMetadata) * instance_metadatas_.size(),
                                        graphics::BUFFER_TYPE_STATIC, &instance_metadata_buffer_);
  }
  instance_metadata_buffer_->UploadData(instance_metadatas_.data(),
                                        sizeof(InstanceMetadata) * instance_metadatas_.size());

  if (!light_selector_buffer_ ||
      sizeof(uint32_t) + sizeof(float) * light_metadatas_.size() > light_selector_buffer_->Size()) {
    core_->GraphicsCore()->CreateBuffer(sizeof(uint32_t) + sizeof(float) * light_metadatas_.size(),
                                        graphics::BUFFER_TYPE_STATIC, &light_selector_buffer_);
  }
  if (!light_metadatas_buffer_ || sizeof(LightMetadata) * light_metadatas_.size() > light_metadatas_buffer_->Size()) {
    core_->GraphicsCore()->CreateBuffer(sizeof(LightMetadata) * light_metadatas_.size(), graphics::BUFFER_TYPE_STATIC,
                                        &light_metadatas_buffer_);
  }
  light_metadatas_buffer_->UploadData(light_metadatas_.data(), sizeof(LightMetadata) * light_metadatas_.size());
  uint32_t light_count = static_cast<uint32_t>(light_metadatas_.size());
  light_selector_buffer_->UploadData(&light_count, sizeof(uint32_t), 0);

  if (buffers_dirty_ || !gather_light_power_program_) {
    gather_light_power_program_.reset();
    core_->GraphicsCore()->CreateComputeProgram(gather_light_power_shader_.get(), &gather_light_power_program_);
    gather_light_power_program_->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, 1);
    gather_light_power_program_->AddResourceBinding(graphics::RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER, 1);
    gather_light_power_program_->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, buffers_.size());
    gather_light_power_program_->Finalize();
  }

  if (hit_groups_dirty_ || callable_shaders_dirty_ || buffers_dirty_ || !rt_program_) {
    auto vfs = core_->GetShadersVFS();
    vfs.WriteFile("geometry_shaders.hlsli", geometry_shader_assembled_);
    vfs.WriteFile("hit_record.hlsli", hit_record_assembled_);
    vfs.WriteFile("material_shaders.hlsli", material_shader_assembled_);
    vfs.WriteFile("light_shaders.hlsli", light_shader_assembled_);
    std::cout << material_shader_assembled_ << std::endl;
    raygen_shader_.reset();
    core_->GraphicsCore()->CreateShader(vfs, "raygen.hlsl", "Main", "lib_6_5", &raygen_shader_);
    rt_program_.reset();
    core_->GraphicsCore()->CreateRayTracingProgram(&rt_program_);
    miss_shader_indices_ = {0};
    callable_shader_indices_.resize(callable_shaders_.size());
    std::iota(callable_shader_indices_.begin(), callable_shader_indices_.end(), 0);
    hit_group_indices_.resize(hit_groups_.size());
    std::iota(hit_group_indices_.begin(), hit_group_indices_.end(), 0);
    rt_program_->AddResourceBinding(graphics::RESOURCE_TYPE_IMAGE, 1);
    rt_program_->AddResourceBinding(graphics::RESOURCE_TYPE_IMAGE, 1);
    rt_program_->AddResourceBinding(graphics::RESOURCE_TYPE_ACCELERATION_STRUCTURE, 1);
    rt_program_->AddResourceBinding(graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);
    rt_program_->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, 1);
    rt_program_->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, 1);
    rt_program_->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, buffers_.size());
    rt_program_->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, 1);
    rt_program_->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, 1);
    rt_program_->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, 1);

    rt_program_->AddRayGenShader(raygen_shader_.get());
    rt_program_->AddMissShader(default_miss_shader_.get());
    for (const auto &hit_group : hit_groups_) {
      rt_program_->AddHitGroup(hit_group);
    }
    for (const auto &callable_shader : callable_shaders_) {
      rt_program_->AddCallableShader(callable_shader);
    }
    rt_program_->Finalize(miss_shader_indices_, hit_group_indices_, callable_shader_indices_);
    hit_groups_dirty_ = false;
    callable_shaders_dirty_ = false;
    buffers_dirty_ = false;
  }

  preprocess_cmd_context_->CmdBindComputeProgram(gather_light_power_program_.get());
  preprocess_cmd_context_->CmdBindResources(0, {light_metadatas_buffer_.get()}, graphics::BIND_POINT_COMPUTE);
  preprocess_cmd_context_->CmdBindResources(1, {light_selector_buffer_.get()}, graphics::BIND_POINT_COMPUTE);
  preprocess_cmd_context_->CmdBindResources(2, buffers_, graphics::BIND_POINT_COMPUTE);
  preprocess_cmd_context_->CmdDispatch((light_metadatas_.size() + 63) / 64, 1, 1);

  blelloch_metadatas_.clear();
  BlellochScanMetadata metadata{4, 4, static_cast<uint32_t>(light_metadatas_.size())};
  uint32_t wave_size = core_->GraphicsCore()->WaveSize();
  while (metadata.element_count > 1) {
    blelloch_metadatas_.push_back(metadata);
    metadata = {metadata.offset + (wave_size - 1) * metadata.stride, metadata.stride * wave_size,
                metadata.element_count / wave_size};
  }
  if (!blelloch_metadatas_.empty()) {
    auto blelloch_scan_up_program = core_->GetComputeProgram("blelloch_scan_up");
    auto blelloch_scan_down_program = core_->GetComputeProgram("blelloch_scan_down");
    if (!blelloch_metadata_buffer_ ||
        sizeof(BlellochScanMetadata) * blelloch_metadatas_.size() > blelloch_metadata_buffer_->Size()) {
      core_->GraphicsCore()->CreateBuffer(sizeof(BlellochScanMetadata) * blelloch_metadatas_.size(),
                                          graphics::BUFFER_TYPE_STATIC, &blelloch_metadata_buffer_);
    }
    blelloch_metadata_buffer_->UploadData(blelloch_metadatas_.data(),
                                          sizeof(BlellochScanMetadata) * blelloch_metadatas_.size());
    preprocess_cmd_context_->CmdBindComputeProgram(blelloch_scan_up_program);
    preprocess_cmd_context_->CmdBindResources(0, {light_selector_buffer_.get()}, graphics::BIND_POINT_COMPUTE);
    for (size_t i = 1; i < blelloch_metadatas_.size(); i++) {
      preprocess_cmd_context_->CmdBindResources(1, {blelloch_metadata_buffer_->Range(sizeof(BlellochScanMetadata) * i)},
                                                graphics::BIND_POINT_COMPUTE);
      preprocess_cmd_context_->CmdDispatch((blelloch_metadatas_[i].element_count + 63) / 64, 1, 1);
    }
    preprocess_cmd_context_->CmdBindComputeProgram(blelloch_scan_down_program);
    for (size_t i = blelloch_metadatas_.size() - 1; i < blelloch_metadatas_.size(); i--) {
      if (blelloch_metadatas_[i].element_count <= wave_size) {
        continue;
      }
      preprocess_cmd_context_->CmdBindResources(1, {blelloch_metadata_buffer_->Range(sizeof(BlellochScanMetadata) * i)},
                                                graphics::BIND_POINT_COMPUTE);
      preprocess_cmd_context_->CmdDispatch((blelloch_metadatas_[i].element_count + 63) / 64, 1, 1);
    }
  }
  core_->GraphicsCore()->SubmitCommandContext(preprocess_cmd_context_.get());
}

}  // namespace sparks
