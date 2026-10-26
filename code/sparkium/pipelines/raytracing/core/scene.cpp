#include "sparkium/pipelines/raytracing/core/scene.h"

#include <numeric>

#include "sparkium/core/frame_profile.h"
#include "sparkium/pipelines/raytracing/core/camera.h"
#include "sparkium/pipelines/raytracing/core/core.h"
#include "sparkium/pipelines/raytracing/core/entity.h"
#include "sparkium/pipelines/raytracing/core/film.h"
#include "sparkium/pipelines/raytracing/core/geometry.h"
#include "sparkium/pipelines/raytracing/core/light.h"
#include "sparkium/pipelines/raytracing/core/material.h"
#include "sparkium/pipelines/raytracing/entity/entities.h"
#include "sparkium/pipelines/raytracing/geometry/geometries.h"
#include "sparkium/pipelines/raytracing/light/lights.h"
#include "sparkium/pipelines/raytracing/material/materials.h"

namespace sparkium::raytracing {

Scene::Scene(sparkium::Scene &scene) : scene_(scene), settings(scene.settings) {
  core_ = DedicatedCast(scene_.GetCore());
  scene_settings_buffer_ = core_->GetDevice()->CreateBuffer(
      {(sizeof(Settings::RayTracing) + sizeof(sparkium::Film::Info) + 255) & ~uint64_t(255)});
  gather_light_power_shader_ =
      core_->GetDevice()->CreateShader("gather_light_power.slang", "GatherLightPowerKernel", rhi::SHADER_STAGE_COMPUTE,
                                       &core_->GetShadersVFS(), {"."}, {});
  linear_sampler_ = core_->GetDevice()->CreateSampler({rhi::SAMPLER_FILTER_LINEAR});
  nearest_sampler_ = core_->GetDevice()->CreateSampler({rhi::SAMPLER_FILTER_NEAREST});
  resolve_group_ =
      core_->GetDevice()->CreateBindingGroup(core_->GetComputeProgram("film2img")->GetDesc().pipeline_layout.groups[0]);
  scan_group_ = core_->GetDevice()->CreateBindingGroup(
      core_->GetComputeProgram("blelloch_scan_up")->GetDesc().pipeline_layout.groups[0]);
}

void Scene::Render(Camera *camera, Film *film, bool software, bool ray_query) {
  software = software || ray_query;
  if (!software && !raygen_shader_) {
    raygen_shader_ = core_->GetDevice()->CreateShader("raygen.slang", "Main", rhi::SHADER_STAGE_RAY_GENERATION,
                                                      &core_->GetShadersVFS(), {"."}, {});
    default_miss_shader_ = core_->GetDevice()->CreateShader("raygen.slang", "MissMain", rhi::SHADER_STAGE_MISS,
                                                            &core_->GetShadersVFS(), {"."}, {});
    shadow_miss_shader_ = core_->GetDevice()->CreateShader("raygen.slang", "ShadowMiss", rhi::SHADER_STAGE_MISS,
                                                           &core_->GetShadersVFS(), {"."}, {});
  }

  if (ray_query != ray_query_) {
    ray_query_ = ray_query;
    software_pipeline_.reset();
    pipeline_dirty_ = true;
    if (rendered_)
      film->Reset();
  }
  if (software != software_tracing_) {
    software_tracing_ = software;
    pipeline_dirty_ = true;
    if (rendered_)
      film->Reset();
  }
  if (software && !software_pipeline_)
    software_pipeline_ = std::make_unique<SoftwarePipeline>(core_, ray_query_);
  CpuProfileScope update_profile("scene_update");
  UpdatePipeline(camera);
  update_profile.End();
  CpuProfileScope setup_profile("render_setup");
  rendered_ = true;
  UploadBuffer(core_->GetDevice(), scene_settings_buffer_.Get(), &settings.raytracing, sizeof(Settings::RayTracing));
  UploadBuffer(core_->GetDevice(), scene_settings_buffer_.Get(), &film->film_.info, sizeof(sparkium::Film::Info),
               sizeof(Settings::RayTracing));
  film->film_.info.accumulated_samples += settings.raytracing.samples_per_dispatch;
  rhi::ObjPtr<rhi::ICommandContext> cmd_context;
  cmd_context = core_->GetDevice()->CreateCommandContext();
  auto *trace_group = software ? software_pipeline_->Bindings() : render_group_.Get();
  if (software)
    cmd_context->CmdBindComputeProgram(software_pipeline_->Program());
  else
    cmd_context->CmdBindRayTracingProgram(rt_program_.Get());
  trace_group->BindResource(0, film->accumulated_color_.Get());
  trace_group->BindResource(1, film->accumulated_samples_.Get());
  if (software) {
    if (ray_query_)
      trace_group->BindResource(2, software_pipeline_->AccelerationStructure());
    else
      trace_group->BindResource(2, software_pipeline_->Nodes());
  } else
    trace_group->BindResource(2, tlas_.Get());
  trace_group->BindResource(3, scene_settings_buffer_.Get());
  if (software) {
    auto resources = buffers_;
    resources.insert(resources.end(),
                     {core_->GetBuffer("sobol"), camera->Buffer(), instance_metadata_buffer_.Get(),
                      light_selector_buffer_.Get(), light_metadatas_buffer_.Get(), software_pipeline_->Instances()});
    trace_group->BindResource(4, resources);
    trace_group->BindResource(5, sdr_images_);
    trace_group->BindResource(6, hdr_images_);
    trace_group->BindResource(7, std::vector{linear_sampler_.Get(), nearest_sampler_.Get()});
  } else {
    trace_group->BindResource(4, core_->GetBuffer("sobol"));
    trace_group->BindResource(5, camera->Buffer());
    trace_group->BindResource(6, buffers_);
    trace_group->BindResource(7, instance_metadata_buffer_.Get());
    trace_group->BindResource(8, light_selector_buffer_.Get());
    trace_group->BindResource(9, light_metadatas_buffer_.Get());
    trace_group->BindResource(10, sdr_images_);
    trace_group->BindResource(11, hdr_images_);
    trace_group->BindResource(12, std::vector{linear_sampler_.Get(), nearest_sampler_.Get()});
  }
  if (software) {
    cmd_context->CmdBindGroup(0, trace_group);
    cmd_context->CmdDispatch((film->GetWidth() + 7) / 8, (film->GetHeight() + 7) / 8, 1);
  } else {
    cmd_context->CmdBindGroup(0, trace_group);
    cmd_context->CmdDispatchRays(film->accumulated_color_->GetDesc().width,
                                 film->accumulated_samples_->GetDesc().height, 1);
  }
  cmd_context->CmdBindComputeProgram(core_->GetComputeProgram("film2img"));
  resolve_group_->BindResource(0, film->accumulated_color_.Get());
  resolve_group_->BindResource(1, film->accumulated_samples_.Get());
  resolve_group_->BindResource(2, film->film_.GetRawImage());
  cmd_context->CmdBindGroup(0, resolve_group_.Get());
  cmd_context->CmdDispatch((film->film_.GetRawImage()->GetDesc().width + 7) / 8,
                           (film->film_.GetRawImage()->GetDesc().height + 7) / 8, 1);
  setup_profile.End();
  CpuProfileScope submit_profile("render_submit_and_wait");
  Execute(core_->GetDevice(), cmd_context.Get());
  submit_profile.End();
}

int32_t Scene::RegisterLight(Light *light, int custom_index) {
  int32_t light_reg_index = light_metadatas_.size();
  LightMetadata light_reg{};
  light_reg.sampler_data_index = RegisterBuffer(light->SamplerData());
  light_reg.sampler_shader_index = light->SamplerShader(this);
  light_reg.custom_index = custom_index;
  light_reg.power_offset = light->SamplerPreprocess(preprocess_cmd_context_.Get());
  light_metadatas_.push_back(light_reg);
  return light_reg_index;
}

int32_t Scene::RegisterInstance(rhi::IBottomLevelAccelerationStructure *blas,
                                const glm::mat4x3 &transformation,
                                int32_t hit_group_index,
                                int32_t geometry_data_index,
                                int32_t material_data_index,
                                int32_t custom_index) {
  int32_t instance_reg = instances_.size();
  InstanceMetadata entity_metadata{};
  rhi::RayTracingInstance instance;
  instance.blas = blas;
  instance.custom_index = geometry_data_index;
  instance.hit_group_offset = hit_group_index * 2;
  for (uint32_t row = 0; row < 3; ++row)
    for (uint32_t column = 0; column < 4; ++column)
      instance.transform[row * 4 + column] = transformation[column][row];
  instances_.push_back(instance);
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

int32_t Scene::RegisterCallableShader(rhi::IShader *callable_shader) {
  if (!callable_shader)
    return -1;
  if (!callable_shader_map_.count(callable_shader)) {
    callable_shader_map_[callable_shader] = static_cast<int32_t>(callable_shaders_.size());
    callable_shaders_.emplace_back(callable_shader);
    pipeline_dirty_ = true;
  }
  return callable_shader_map_[callable_shader];
}

int32_t Scene::RegisterBuffer(rhi::IBuffer *buffer) {
  if (!buffer)
    return -1;
  if (!buffer_map_.count(buffer)) {
    buffer_map_[buffer] = static_cast<int32_t>(buffers_.size());
    buffers_.emplace_back(buffer);
  }
  return buffer_map_[buffer];
}

int32_t Scene::RegisterImage(rhi::IImage *image) {
  if (image->GetDesc().format == rhi::IMAGE_FORMAT_R8G8B8A8_UNORM) {
    if (!sdr_image_map_.count(image)) {
      sdr_image_map_[image] = static_cast<int32_t>(sdr_images_.size());
      sdr_images_.emplace_back(image);
    }
    return sdr_image_map_[image];
  }
  if (image->GetDesc().format == rhi::IMAGE_FORMAT_R32G32B32A32_SFLOAT) {
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
  if (!software_tracing_ && (callable_shaders_.empty() || callable_shaders_[0] != camera->Shader()))
    pipeline_dirty_ = true;
  CpuProfileScope registration_profile("scene_registration");
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

  preprocess_cmd_context_.Release();
  preprocess_cmd_context_ = core_->GetDevice()->CreateCommandContext();

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
    if (!software_tracing_)
      RegisterCallableShader(camera->Shader());
  }
  for (auto *entity : ordered_entities) {
    if (entities_.at(entity).active) {
      entity->Update(this);
    }
  }
  registration_profile.End();
  CpuProfileScope metadata_profile("scene_metadata");
  if (!software_tracing_) {
    if (!tlas_ || tlas_->GetDesc().instance_count < instances_.size())
      tlas_ = core_->GetDevice()->CreateTopLevelAccelerationStructure(
          {static_cast<uint32_t>(std::max<size_t>(1, instances_.size()))});
    if (!tlas_ || !preprocess_cmd_context_->CmdBuildTopLevelAccelerationStructure(tlas_.Get(), instances_))
      throw std::runtime_error("failed to build ray tracing TLAS");
  }

  if (!instance_metadata_buffer_ ||
      sizeof(InstanceMetadata) * instance_metadatas_.size() > instance_metadata_buffer_->GetDesc().size) {
    instance_metadata_buffer_.Release();
    instance_metadata_buffer_ =
        core_->GetDevice()->CreateBuffer({sizeof(InstanceMetadata) * std::max<size_t>(1, instance_metadatas_.size())});
  }
  if (!instance_metadatas_.empty())
    UploadBuffer(core_->GetDevice(), instance_metadata_buffer_.Get(), instance_metadatas_.data(),
                 sizeof(InstanceMetadata) * instance_metadatas_.size());

  if (!light_selector_buffer_ ||
      sizeof(uint32_t) + sizeof(float) * light_metadatas_.size() > light_selector_buffer_->GetDesc().size) {
    light_selector_buffer_ =
        core_->GetDevice()->CreateBuffer({sizeof(uint32_t) + sizeof(float) * light_metadatas_.size()});
  }
  if (!light_metadatas_buffer_ ||
      sizeof(LightMetadata) * light_metadatas_.size() > light_metadatas_buffer_->GetDesc().size) {
    light_metadatas_buffer_ =
        core_->GetDevice()->CreateBuffer({sizeof(LightMetadata) * std::max<size_t>(1, light_metadatas_.size())});
  }
  if (!light_metadatas_.empty())
    UploadBuffer(core_->GetDevice(), light_metadatas_buffer_.Get(), light_metadatas_.data(),
                 sizeof(LightMetadata) * light_metadatas_.size());
  uint32_t light_count = static_cast<uint32_t>(light_metadatas_.size());
  UploadBuffer(core_->GetDevice(), light_selector_buffer_.Get(), &light_count, sizeof(uint32_t), 0);

  if (buffers_.empty())
    RegisterBuffer(camera->Buffer());

  if (pipeline_dirty_ || buffers_.size() > buffer_capacity_ || sdr_images_.size() > sdr_image_capacity_ ||
      hdr_images_.size() > hdr_image_capacity_ || (!software_tracing_ && !rt_program_)) {
    if (!software_tracing_) {
      rhi::RayTracingProgramDesc desc;
      desc.ray_generation = raygen_shader_.Get();
      desc.miss_shaders = {default_miss_shader_.Get(), shadow_miss_shader_.Get()};
      // RenderContext in common.slang is 39 scalar words; triangle attributes are float2.
      desc.max_payload_size = 39 * sizeof(uint32_t);
      desc.max_recursion_depth = 2;
      for (const auto &group : hit_groups_) {
        desc.hit_groups.push_back(group.render_group);
        desc.hit_groups.push_back(group.shadow_group);
      }
      desc.callable_shaders = callable_shaders_;
      desc.pipeline_layout.groups = {{{{0, rhi::BINDING_RW_TEXTURE2D},
                                       {1, rhi::BINDING_RW_TEXTURE2D},
                                       {2, rhi::BINDING_ACCELERATION_STRUCTURE},
                                       {3, rhi::BINDING_UNIFORM_BUFFER},
                                       {4, rhi::BINDING_BYTE_ADDRESS_BUFFER},
                                       {5, rhi::BINDING_BYTE_ADDRESS_BUFFER},
                                       {6, rhi::BINDING_BYTE_ADDRESS_BUFFER, static_cast<uint32_t>(buffers_.size())},
                                       {7, rhi::BINDING_BYTE_ADDRESS_BUFFER},
                                       {8, rhi::BINDING_BYTE_ADDRESS_BUFFER},
                                       {9, rhi::BINDING_BYTE_ADDRESS_BUFFER},
                                       {10, rhi::BINDING_TEXTURE2D, static_cast<uint32_t>(sdr_images_.size())},
                                       {11, rhi::BINDING_TEXTURE2D, static_cast<uint32_t>(hdr_images_.size())},
                                       {12, rhi::BINDING_SAMPLER, 2}}}};
      rt_program_ = core_->GetDevice()->CreateRayTracingProgram(desc);
      if (!rt_program_)
        throw std::runtime_error("failed to create ray tracing pipeline");
      render_group_ = core_->GetDevice()->CreateBindingGroup(desc.pipeline_layout.groups[0]);
    }

    if (buffers_.size() > buffer_capacity_ || !gather_light_power_program_) {
      gather_light_power_program_.Release();
      {
        rhi::ComputeProgramDesc desc;
        desc.compute_shader = gather_light_power_shader_.Get();
        desc.pipeline_layout.groups = {
            {{{0, rhi::BINDING_BYTE_ADDRESS_BUFFER, 1},
              {1, rhi::BINDING_RW_BYTE_ADDRESS_BUFFER, 1},
              {2, rhi::BINDING_BYTE_ADDRESS_BUFFER, static_cast<uint32_t>(buffers_.size())}}}};
        gather_light_power_program_ = core_->GetDevice()->CreateComputeProgram(desc);
      }
      light_group_ =
          core_->GetDevice()->CreateBindingGroup(gather_light_power_program_->GetDesc().pipeline_layout.groups[0]);
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
    software_pipeline_->Update(preprocess_cmd_context_.Get(), buffers_, sdr_images_.size(), hdr_images_.size(), camera);

  CpuProfileScope light_record_profile("light_selection_record");
  preprocess_cmd_context_->CmdBindComputeProgram(gather_light_power_program_.Get());
  light_group_->BindResource(0, light_metadatas_buffer_.Get());
  light_group_->BindResource(1, light_selector_buffer_.Get());
  light_group_->BindResource(2, buffers_);
  if (!light_metadatas_.empty()) {
    preprocess_cmd_context_->CmdBindGroup(0, light_group_.Get());
    preprocess_cmd_context_->CmdDispatch((light_metadatas_.size() + 63) / 64, 1, 1);
  }

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
        sizeof(BlellochScanMetadata) * blelloch_metadatas_.size() > blelloch_metadata_buffer_->GetDesc().size) {
      blelloch_metadata_buffer_ =
          core_->GetDevice()->CreateBuffer({sizeof(BlellochScanMetadata) * blelloch_metadatas_.size()});
    }
    UploadBuffer(core_->GetDevice(), blelloch_metadata_buffer_.Get(), blelloch_metadatas_.data(),
                 sizeof(BlellochScanMetadata) * blelloch_metadatas_.size());
    preprocess_cmd_context_->CmdBindComputeProgram(blelloch_scan_up_program);
    scan_group_->BindResource(0, light_selector_buffer_.Get());
    for (size_t i = 1; i < blelloch_metadatas_.size(); i++) {
      scan_group_->BindResource(1, rhi::BufferView{blelloch_metadata_buffer_.Get(), sizeof(BlellochScanMetadata) * i,
                                                   sizeof(BlellochScanMetadata)});
      preprocess_cmd_context_->CmdBindGroup(0, scan_group_.Get());
      preprocess_cmd_context_->CmdDispatch((blelloch_metadatas_[i].element_count + 63) / 64, 1, 1);
    }
    preprocess_cmd_context_->CmdBindComputeProgram(blelloch_scan_down_program);
    for (size_t i = blelloch_metadatas_.size() - 1; i < blelloch_metadatas_.size(); i--) {
      if (blelloch_metadatas_[i].element_count <= group_size) {
        continue;
      }
      scan_group_->BindResource(1, rhi::BufferView{blelloch_metadata_buffer_.Get(), sizeof(BlellochScanMetadata) * i,
                                                   sizeof(BlellochScanMetadata)});
      preprocess_cmd_context_->CmdBindGroup(0, scan_group_.Get());
      preprocess_cmd_context_->CmdDispatch((blelloch_metadatas_[i].element_count + 63) / 64, 1, 1);
    }
  }
  light_record_profile.End();
  CpuProfileScope preprocess_submit_profile("preprocess_submit_and_wait");
  Execute(core_->GetDevice(), preprocess_cmd_context_.Get());
}

Scene *DedicatedCast(sparkium::Scene *scene) {
  COMPONENT_CAST(scene, Scene);
}

}  // namespace sparkium::raytracing
