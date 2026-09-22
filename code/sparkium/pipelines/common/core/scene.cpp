#include "sparkium/pipelines/common/core/scene.h"

#include <numeric>

#include "grassland/graphics/frame_profile.h"
#include "sparkium/pipelines/common/core/camera.h"
#include "sparkium/pipelines/common/core/core.h"
#include "sparkium/pipelines/common/core/entity.h"
#include "sparkium/pipelines/common/core/geometry.h"
#include "sparkium/pipelines/common/core/light.h"
#include "sparkium/pipelines/common/core/material.h"
#include "sparkium/pipelines/common/entity/entities.h"
#include "sparkium/pipelines/common/geometry/geometries.h"
#include "sparkium/pipelines/common/light/lights.h"
#include "sparkium/pipelines/common/material/materials.h"

namespace sparkium::render_shared {

Scene::Scene(sparkium::Scene &scene) : scene_(scene), settings(scene.settings) {
  core_ = DedicatedCast(scene_.GetCore());
  if (core_->GraphicsCore()->DeviceRayTracingSupport()) {
    core_->GraphicsCore()->CreateShader(core_->GetShadersVFS(), "raygen.hlsl", "Main", "lib_6_5", &raygen_shader_);
    core_->GraphicsCore()->CreateShader(core_->GetShadersVFS(), "raygen.hlsl", "MissMain", "lib_6_5",
                                        &default_miss_shader_);
    core_->GraphicsCore()->CreateShader(core_->GetShadersVFS(), "raygen.hlsl", "ShadowMiss", "lib_6_5",
                                        &shadow_miss_shader_);
  }
  core_->GraphicsCore()->CreateBuffer(sizeof(Settings::RayTracing) + sizeof(sparkium::Film::Info),
                                      graphics::BUFFER_TYPE_STATIC, &scene_settings_buffer_);
  core_->GraphicsCore()->CreateShader(core_->GetShadersVFS(), "gather_light_power.hlsl", "GatherLightPowerKernel",
                                      "cs_6_3", &gather_light_power_shader_);
  core_->GraphicsCore()->CreateSampler({graphics::FILTER_MODE_LINEAR}, &linear_sampler_);
  core_->GraphicsCore()->CreateSampler({graphics::FILTER_MODE_NEAREST}, &nearest_sampler_);
}

int32_t Scene::RegisterLight(Light *light, int custom_index) {
  int32_t light_reg_index = light_metadatas_.size();
  LightMetadata light_reg{};
  light_reg.sampler_data_index = RegisterBuffer(light->SamplerData());
  light_reg.sampler_shader_index = light->SamplerShader(this);
  light_reg.custom_index = custom_index;
  light_reg.power_offset = light->SamplerPreprocess(preprocess_cmd_context_.get());
  light_metadatas_.push_back(light_reg);
  return light_reg_index;
}

int32_t Scene::RegisterInstance(graphics::AccelerationStructure *blas,
                                const glm::mat4x3 &transformation,
                                int32_t hit_group_index,
                                int32_t geometry_data_index,
                                int32_t material_data_index,
                                int32_t custom_index) {
  int32_t instance_reg = instances_.size();
  InstanceMetadata entity_metadata{};
  instances_.push_back(blas->MakeInstance(transformation, geometry_data_index, 255, hit_group_index * 2));
  entity_metadata.geometry_data_index = geometry_data_index;
  entity_metadata.material_data_index = material_data_index;
  entity_metadata.custom_index = custom_index;
  instance_metadatas_.push_back(entity_metadata);
  return instance_reg;
}

int32_t Scene::RegisterSoftwareInstance(Geometry *geometry,
                                        Material *material,
                                        const glm::mat4x3 &transform,
                                        int32_t custom_index) {
  const int32_t index = instance_metadatas_.size();
  const int32_t geometry_index = RegisterBuffer(geometry->Buffer());
  instance_metadatas_.push_back({geometry_index, RegisterBuffer(material->Buffer()), custom_index});
  software_pipeline_->AddInstance(geometry, material, transform, geometry_index);
  return index;
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
    pipeline_dirty_ = true;
  }
  return callable_shader_map_[callable_shader];
}

int32_t Scene::RegisterBuffer(graphics::Buffer *buffer) {
  if (!buffer)
    return -1;
  if (!buffer_map_.count(buffer)) {
    buffer_map_[buffer] = static_cast<int32_t>(buffers_.size());
    buffers_.emplace_back(buffer);
  }
  return buffer_map_[buffer];
}

int32_t Scene::RegisterImage(graphics::Image *image) {
  if (image->Format() == graphics::IMAGE_FORMAT_R8G8B8A8_UNORM) {
    if (!sdr_image_map_.count(image)) {
      sdr_image_map_[image] = static_cast<int32_t>(sdr_images_.size());
      sdr_images_.emplace_back(image);
    }
    return sdr_image_map_[image];
  }
  if (image->Format() == graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT) {
    if (!hdr_image_map_.count(image)) {
      hdr_image_map_[image] = static_cast<int32_t>(hdr_images_.size());
      hdr_images_.emplace_back(image);
    }
    return hdr_image_map_[image] + 0x1000000;
  }
  return -1;
}

int32_t Scene::RegisterHitGroup(const InstanceHitGroups &hit_group) {
  if (!hit_group_map_.count(hit_group)) {
    hit_group_map_[hit_group] = static_cast<int32_t>(hit_groups_.size());
    hit_groups_.emplace_back(hit_group);
    pipeline_dirty_ = true;
  }
  return hit_group_map_[hit_group];
}

void Scene::UpdatePipeline(Camera *camera) {
  graphics::CpuProfileScope registration_profile("scene_registration");
  for (auto &[entity, status] : entities_) {
    status.keep = false;
  }

  std::vector<Entity *> ordered_entities;
  for (auto *entity : scene_.GetEntityOrder()) {
    const auto &status = scene_.GetEntities().at(entity);
    auto rt_entity = DedicatedCast(entity);
    if (rt_entity) {
      entities_[rt_entity].keep = true;
      entities_[rt_entity].active = status.active;
      ordered_entities.push_back(rt_entity);
    }
  }

  std::set<Entity *> to_delete_entities;

  for (auto &[entity, status] : entities_) {
    if (!status.keep) {
      pipeline_dirty_ = true;
      to_delete_entities.insert(entity);
    }
  }

  for (auto entity : to_delete_entities) {
    entities_.erase(entity);
  }

  if (software_tracing_)
    software_pipeline_->ClearInstances();
  instances_.clear();
  instance_metadatas_.clear();
  light_metadatas_.clear();

  preprocess_cmd_context_.reset();
  core_->GraphicsCore()->CreateCommandContext(&preprocess_cmd_context_);

  bool &pipeline_dirty = pipeline_dirty_;

  buffers_.clear();
  buffer_map_.clear();

  sdr_images_.clear();
  sdr_image_map_.clear();

  hdr_images_.clear();
  hdr_image_map_.clear();
  RegisterImage(core_->GetImage("white"));
  RegisterImage(core_->GetImage("white_hdr"));

  if (pipeline_dirty) {
    hit_groups_.clear();
    hit_group_map_.clear();
    callable_shaders_.clear();
    callable_shader_map_.clear();
    RegisterCallableShader(camera->Shader());
  }

  graphics::GpuProfileScope geometry_light_profile(preprocess_cmd_context_.get(), "light_geometry");
  for (auto *entity : ordered_entities) {
    if (entities_.at(entity).active) {
      entity->Update(this);
    }
  }

  geometry_light_profile.End();
  registration_profile.End();
  graphics::CpuProfileScope metadata_profile("scene_metadata");
  if (!software_tracing_) {
    if (!tlas_) {
      core_->GraphicsCore()->CreateTopLevelAccelerationStructure(instances_, &tlas_);
    } else {
      tlas_->UpdateInstances(instances_);
    }
  }

  if (!instance_metadata_buffer_ ||
      sizeof(InstanceMetadata) * instance_metadatas_.size() > instance_metadata_buffer_->Size()) {
    instance_metadata_buffer_.reset();
    core_->GraphicsCore()->CreateBuffer(sizeof(InstanceMetadata) * std::max<size_t>(1, instance_metadatas_.size()),
                                        graphics::BUFFER_TYPE_STATIC, &instance_metadata_buffer_);
  }
  if (!instance_metadatas_.empty())
    instance_metadata_buffer_->UploadData(instance_metadatas_.data(),
                                          sizeof(InstanceMetadata) * instance_metadatas_.size());

  if (!light_selector_buffer_ ||
      sizeof(uint32_t) + sizeof(float) * light_metadatas_.size() > light_selector_buffer_->Size()) {
    core_->GraphicsCore()->CreateBuffer(sizeof(uint32_t) + sizeof(float) * light_metadatas_.size(),
                                        graphics::BUFFER_TYPE_STATIC, &light_selector_buffer_);
  }
  if (!light_metadatas_buffer_ || sizeof(LightMetadata) * light_metadatas_.size() > light_metadatas_buffer_->Size()) {
    core_->GraphicsCore()->CreateBuffer(sizeof(LightMetadata) * std::max<size_t>(1, light_metadatas_.size()),
                                        graphics::BUFFER_TYPE_STATIC, &light_metadatas_buffer_);
  }
  if (!light_metadatas_.empty())
    light_metadatas_buffer_->UploadData(light_metadatas_.data(), sizeof(LightMetadata) * light_metadatas_.size());
  uint32_t light_count = static_cast<uint32_t>(light_metadatas_.size());
  light_selector_buffer_->UploadData(&light_count, sizeof(uint32_t), 0);

  if (buffers_.empty())
    RegisterBuffer(camera->Buffer());

  if (pipeline_dirty_ || buffers_.size() > buffer_capacity_ || sdr_images_.size() > sdr_image_capacity_ ||
      hdr_images_.size() > hdr_image_capacity_ || (!software_tracing_ && !rt_program_)) {
    if (!software_tracing_) {
      rt_program_.reset();
      core_->GraphicsCore()->CreateRayTracingProgram(&rt_program_);
      miss_shader_indices_ = {0, 1};
      callable_shader_indices_.resize(callable_shaders_.size());
      std::iota(callable_shader_indices_.begin(), callable_shader_indices_.end(), 0);
      hit_group_indices_.resize(hit_groups_.size() * 2);
      std::iota(hit_group_indices_.begin(), hit_group_indices_.end(), 0);
      rt_program_->AddResourceBinding(graphics::RESOURCE_TYPE_WRITABLE_IMAGE, 1);
      rt_program_->AddResourceBinding(graphics::RESOURCE_TYPE_WRITABLE_IMAGE, 1);
      rt_program_->AddResourceBinding(graphics::RESOURCE_TYPE_ACCELERATION_STRUCTURE, 1);
      rt_program_->AddResourceBinding(graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);
      rt_program_->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, 1);
      rt_program_->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, 1);
      rt_program_->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, buffers_.size());
      rt_program_->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, 1);
      rt_program_->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, 1);
      rt_program_->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, 1);
      rt_program_->AddResourceBinding(graphics::RESOURCE_TYPE_IMAGE, sdr_images_.size());
      rt_program_->AddResourceBinding(graphics::RESOURCE_TYPE_IMAGE, hdr_images_.size());
      rt_program_->AddResourceBinding(graphics::RESOURCE_TYPE_SAMPLER, 2);

      rt_program_->AddRayGenShader(raygen_shader_.get());
      rt_program_->AddMissShader(default_miss_shader_.get());
      rt_program_->AddMissShader(shadow_miss_shader_.get());
      for (const auto &hit_group : hit_groups_) {
        rt_program_->AddHitGroup(hit_group.render_group);
        rt_program_->AddHitGroup(hit_group.shadow_group);
      }
      for (const auto &callable_shader : callable_shaders_) {
        rt_program_->AddCallableShader(callable_shader);
      }
      rt_program_->Finalize(miss_shader_indices_, hit_group_indices_, callable_shader_indices_);
    }

    if (buffers_.size() > buffer_capacity_ || !gather_light_power_program_) {
      gather_light_power_program_.reset();
      core_->GraphicsCore()->CreateComputeProgram(gather_light_power_shader_.get(), &gather_light_power_program_);
      gather_light_power_program_->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, 1);
      gather_light_power_program_->AddResourceBinding(graphics::RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER, 1);
      gather_light_power_program_->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, buffers_.size());
      gather_light_power_program_->Finalize();
    }

    buffer_capacity_ = buffers_.size();
    sdr_image_capacity_ = sdr_images_.size();
    hdr_image_capacity_ = hdr_images_.size();
    pipeline_dirty_ = false;
  }

  while (buffers_.size() < buffer_capacity_) {
    buffers_.push_back(camera->Buffer());
  }

  while (sdr_images_.size() < sdr_image_capacity_) {
    sdr_images_.push_back(core_->GetImage("white"));
  }

  while (hdr_images_.size() < hdr_image_capacity_) {
    hdr_images_.push_back(core_->GetImage("white_hdr"));
  }

  metadata_profile.End();
  if (software_tracing_)
    software_pipeline_->Update(preprocess_cmd_context_.get(), buffers_, sdr_images_.size(), hdr_images_.size());

  graphics::CpuProfileScope light_record_profile("light_selection_record");
  graphics::GpuProfileScope light_selection_profile(preprocess_cmd_context_.get(), "light_selection");
  preprocess_cmd_context_->CmdBindComputeProgram(gather_light_power_program_.get());
  preprocess_cmd_context_->CmdBindResources(0, {light_metadatas_buffer_.get()}, graphics::BIND_POINT_COMPUTE);
  preprocess_cmd_context_->CmdBindResources(1, {light_selector_buffer_.get()}, graphics::BIND_POINT_COMPUTE);
  preprocess_cmd_context_->CmdBindResources(2, buffers_, graphics::BIND_POINT_COMPUTE);
  if (!light_metadatas_.empty())
    preprocess_cmd_context_->CmdDispatch((light_metadatas_.size() + 63) / 64, 1, 1);

  blelloch_metadatas_.clear();
  BlellochScanMetadata metadata{4, 4, static_cast<uint32_t>(light_metadatas_.size())};
  uint32_t group_size = 64;
  while (metadata.element_count > 1) {
    blelloch_metadatas_.push_back(metadata);
    metadata = {metadata.offset + (group_size - 1) * metadata.stride, metadata.stride * group_size,
                metadata.element_count / group_size};
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
      if (blelloch_metadatas_[i].element_count <= group_size) {
        continue;
      }
      preprocess_cmd_context_->CmdBindResources(1, {blelloch_metadata_buffer_->Range(sizeof(BlellochScanMetadata) * i)},
                                                graphics::BIND_POINT_COMPUTE);
      preprocess_cmd_context_->CmdDispatch((blelloch_metadatas_[i].element_count + 63) / 64, 1, 1);
    }
  }
  light_selection_profile.End();
  light_record_profile.End();
  graphics::CpuProfileScope preprocess_submit_profile("preprocess_submit");
  core_->GraphicsCore()->SubmitCommandContext(preprocess_cmd_context_.get());
}

}  // namespace sparkium::render_shared
