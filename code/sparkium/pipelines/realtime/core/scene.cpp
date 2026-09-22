#include "sparkium/pipelines/realtime/core/scene.h"

#include <numeric>

#include "grassland/graphics/frame_profile.h"
#include "sparkium/pipelines/realtime/core/camera.h"
#include "sparkium/pipelines/realtime/core/core.h"
#include "sparkium/pipelines/realtime/core/entity.h"
#include "sparkium/pipelines/realtime/core/geometry.h"
#include "sparkium/pipelines/realtime/core/light.h"
#include "sparkium/pipelines/realtime/core/material.h"
#include "sparkium/pipelines/realtime/core/realtime_view.h"
#include "sparkium/pipelines/realtime/entity/entities.h"
#include "sparkium/pipelines/realtime/geometry/geometries.h"
#include "sparkium/pipelines/realtime/light/lights.h"
#include "sparkium/pipelines/realtime/material/materials.h"

namespace sparkium::realtime {
namespace {
RealtimeView *View(sparkium::Film *film) {
  COMPONENT_CAST(film, RealtimeView);
}
}  // namespace

Scene::Scene(sparkium::Scene &scene) : scene_(scene), settings(scene.settings) {
  core_ = DedicatedCast(scene_.GetCore());
  software_pipeline_ = std::make_unique<SoftwarePipeline>(core_);
  core_->GraphicsCore()->CreateBuffer(sizeof(Settings::RayTracing) + sizeof(sparkium::Film::Info),
                                      graphics::BUFFER_TYPE_STATIC, &scene_settings_buffer_);
  core_->GraphicsCore()->CreateShader(core_->GetShadersVFS(), "gather_light_power.hlsl", "GatherLightPowerKernel",
                                      "cs_6_3", &gather_light_power_shader_);
  core_->GraphicsCore()->CreateSampler({graphics::FILTER_MODE_LINEAR}, &linear_sampler_);
  core_->GraphicsCore()->CreateSampler({graphics::FILTER_MODE_NEAREST}, &nearest_sampler_);
}

void Scene::Render(sparkium::Camera *source_camera, sparkium::Film *film) {
  auto *core = core_;
  auto *camera = DedicatedCast(source_camera);
  auto *view = View(film);
  auto *pipeline = software_pipeline_.get();
  const auto &settings = scene_.settings;
  graphics::CpuProfileScope update_profile("scene_update");
  const uint64_t key = RealtimeKey();
  const bool scene_changed = !rendered_ || realtime_key_ != key || pipeline_dirty_;
  if (scene_changed || film->info.accumulated_samples == 0) {
    UpdatePipeline(camera);
    if (scene_changed)
      film->Reset();
    realtime_key_ = key;
  }
  update_profile.End();
  graphics::CpuProfileScope setup_profile("render_setup");
  rendered_ = true;
  auto trace_settings = settings.raytracing;
  trace_settings.samples_per_dispatch = 1;
  trace_settings.max_bounces = std::clamp(settings.realtime.bounces, 1, 8);
  const uint32_t frame = film->info.accumulated_samples;
  scene_settings_buffer_->UploadData(&trace_settings, sizeof(trace_settings));
  scene_settings_buffer_->UploadData(&film->info, sizeof(sparkium::Film::Info), sizeof(trace_settings));
  ++film->info.accumulated_samples;
  std::unique_ptr<graphics::CommandContext> commands;
  core->GraphicsCore()->CreateCommandContext(&commands);
  view->Begin(commands.get(), pipeline, buffers_, camera, film->GetWidth(), film->GetHeight(), settings.realtime.scale,
              settings.realtime.history, settings.realtime.updates, frame);
  graphics::GpuProfileScope trace_profile(commands.get(), "realtime_lighting");
  for (int phase = 0; phase < 2; ++phase) {
    commands->CmdBindComputeProgram(phase == 0 ? pipeline->ReprojectProgram() : pipeline->Program());
    constexpr auto bind_point = graphics::BIND_POINT_COMPUTE;
    commands->CmdBindResources(2, {pipeline->Nodes()}, bind_point);
    commands->CmdBindResources(3, {scene_settings_buffer_.get()}, bind_point);
    auto resources = buffers_;
    resources.insert(resources.end(), {core->GetBuffer("sobol"), camera->Buffer(), instance_metadata_buffer_.get(),
                                       light_selector_buffer_.get(), light_metadatas_buffer_.get(),
                                       pipeline->Instances(), view->ParametersBuffer()});
    commands->CmdBindResources(4, resources, bind_point);
    commands->CmdBindResources(5, sdr_images_, bind_point);
    commands->CmdBindResources(6, hdr_images_, bind_point);
    commands->CmdBindResources(7, std::vector{linear_sampler_.get(), nearest_sampler_.get()}, bind_point);
    view->BindTrace(commands.get());
    uint32_t trace_width = view->Width();
    if (phase == 1) {
      const uint32_t updates = std::clamp(settings.realtime.updates, 1, 16);
      trace_width = (trace_width + updates - 1) / updates;
    }
    commands->CmdDispatch((trace_width + 7) / 8, (view->Height() + 7) / 8, 1);
  }
  trace_profile.End();
  graphics::GpuProfileScope resolve_profile(commands.get(), "film_resolve");
  view->Resolve(commands.get(), film->GetRawImage(), pipeline, buffers_);
  resolve_profile.End();
  setup_profile.End();
  graphics::CpuProfileScope submit_profile("render_submit");
  core->GraphicsCore()->SubmitCommandContext(commands.get());
  submit_profile.End();
  graphics::CpuProfileScope wait_profile("render_wait");
  core->GraphicsCore()->WaitGPU();
}

uint64_t Scene::RealtimeKey() const {
  uint64_t key = 1469598103934665603ull;
  auto add = [&](const auto &value) {
    const auto *bytes = reinterpret_cast<const uint8_t *>(&value);
    for (size_t i = 0; i < sizeof(value); ++i)
      key = (key ^ bytes[i]) * 1099511628211ull;
  };
  add(scene_.settings.raytracing);
  add(scene_.settings.realtime);
  for (auto *entity : scene_.GetEntityOrder()) {
    add(entity);
    add(scene_.GetEntities().at(entity).active);
    if (auto *instance = dynamic_cast<sparkium::EntityGeometryMaterial *>(entity)) {
      add(instance->transform);
      auto *geometry = instance->GetGeometry();
      auto *material = instance->GetMaterial();
      add(geometry);
      add(material);
      if (auto *m = dynamic_cast<sparkium::MaterialLambertian *>(material)) {
        add(m->base_color);
        add(m->emission);
      }
      if (auto *m = dynamic_cast<sparkium::MaterialPrincipled *>(material)) {
        add(m->info);
        add(m->textures);
      }
      if (auto *m = dynamic_cast<sparkium::MaterialSpecular *>(material))
        add(m->base_color);
      if (auto *m = dynamic_cast<sparkium::MaterialLight *>(material)) {
        add(m->emission);
        add(m->block_ray);
        add(m->camera_visible);
        add(m->two_sided);
        add(m->falloff_distance);
      }
      if (auto *m = dynamic_cast<sparkium::MaterialShaderGraph *>(material))
        add(m->emission_hint);
    }
    if (auto *light = dynamic_cast<sparkium::EntityPointLight *>(entity)) {
      add(light->position);
      add(light->color);
      add(light->strength);
      add(light->radius);
      add(light->soft_falloff);
      add(light->sampling_weight);
    }
  }
  return key;
}

int32_t Scene::RegisterLight(Light *light, int custom_index) {
  int32_t light_reg_index = light_metadatas_.size();
  LightMetadata light_reg{};
  light_reg.sampler_data_index = RegisterBuffer(light->SamplerData());
  light_reg.sampler_shader_index = light->SamplerKind();
  light_reg.custom_index = custom_index;
  light_reg.power_offset = light->SamplerPreprocess(preprocess_cmd_context_.get());
  light_metadatas_.push_back(light_reg);
  return light_reg_index;
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

  software_pipeline_->ClearInstances();
  instance_metadatas_.clear();
  light_metadatas_.clear();

  preprocess_cmd_context_.reset();
  core_->GraphicsCore()->CreateCommandContext(&preprocess_cmd_context_);

  buffers_.clear();
  buffer_map_.clear();

  sdr_images_.clear();
  sdr_image_map_.clear();

  hdr_images_.clear();
  hdr_image_map_.clear();
  RegisterImage(core_->GetImage("white"));
  RegisterImage(core_->GetImage("white_hdr"));

  graphics::GpuProfileScope geometry_light_profile(preprocess_cmd_context_.get(), "light_geometry");
  for (auto *entity : ordered_entities) {
    if (entities_.at(entity).active) {
      entity->Update(this);
    }
  }

  geometry_light_profile.End();
  registration_profile.End();
  graphics::CpuProfileScope metadata_profile("scene_metadata");

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
      hdr_images_.size() > hdr_image_capacity_) {
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

}  // namespace sparkium::realtime
