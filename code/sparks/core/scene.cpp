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
  core_->GraphicsCore()->CreateShader(core_->GetShadersVFS(), "gather_light_power.hlsl", "GatherLightPowerKernel",
                                      "cs_6_3", &gather_light_power_shader_);
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
  cmd_context->CmdBindResources(6, {instance_metadata_buffer_.get()}, graphics::BIND_POINT_RAYTRACING);
  cmd_context->CmdBindResources(7, {light_selector_buffer_.get()}, graphics::BIND_POINT_RAYTRACING);
  cmd_context->CmdBindResources(8, {light_metadatas_buffer_.get()}, graphics::BIND_POINT_RAYTRACING);
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

int32_t Scene::RegisterLight(Light *light, int custom_index) {
  int32_t light_reg_index = light_metadatas_.size();
  LightMetadata light_reg{};
  light_reg.sampler_data_index = RegisterBuffer(light->SamplerData());
  light_reg.sampler_shader_index = RegisterCallableShader(light->SamplerShader());
  light_reg.custom_index = custom_index;
  light_reg.power_offset = light->SamplerPreprocess(preprocess_cmd_context_.get());
  light_metadatas_.push_back(light_reg);
  return light_reg_index;
}

int32_t Scene::RegisterInstance(GeometryRegistration geom_reg,
                                const glm::mat4x3 &transformation,
                                SurfaceRegistration surf_reg,
                                int custom_index) {
  int32_t instance_reg = instances_.size();
  InstanceMetadata entity_metadata{};
  entity_metadata.geometry_data_index = geom_reg.data_index;
  entity_metadata.surface_data_index = surf_reg.data_index;
  entity_metadata.surface_shader_index = surf_reg.shader_index;
  entity_metadata.custom_index = custom_index;
  instances_.push_back(geom_reg.blas->MakeInstance(transformation, geom_reg.data_index, 255, geom_reg.hit_group_index));
  instance_metadatas_.push_back(entity_metadata);
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

int32_t Scene::RegisterHitGroup(const graphics::HitGroup &hit_group) {
  if (!hit_group_map_.count(hit_group)) {
    hit_group_map_[hit_group] = static_cast<int32_t>(hit_groups_.size());
    hit_groups_.emplace_back(hit_group);
  }
  return hit_group_map_[hit_group];
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
  instance_metadatas_.clear();
  light_selector_buffer_.reset();
  light_metadatas_.clear();
  light_metadatas_buffer_.reset();

  blelloch_metadatas_.clear();
  blelloch_metadata_buffer_.reset();

  gather_light_power_program_.reset();

  RegisterCallableShader(camera->Shader());

  preprocess_cmd_context_.reset();
  core_->GraphicsCore()->CreateCommandContext(&preprocess_cmd_context_);

  for (auto entity : entities_) {
    entity->Update(this);
  }

  callable_shader_indices_.resize(callable_shaders_.size());
  std::iota(callable_shader_indices_.begin(), callable_shader_indices_.end(), 0);
  hit_group_indices_.resize(hit_groups_.size());
  std::iota(hit_group_indices_.begin(), hit_group_indices_.end(), 0);

  core_->GraphicsCore()->CreateTopLevelAccelerationStructure(instances_, &tlas_);

  instance_metadata_buffer_.reset();
  core_->GraphicsCore()->CreateBuffer(sizeof(InstanceMetadata) * instance_metadatas_.size(),
                                      graphics::BUFFER_TYPE_STATIC, &instance_metadata_buffer_);
  instance_metadata_buffer_->UploadData(instance_metadatas_.data(),
                                        sizeof(InstanceMetadata) * instance_metadatas_.size());

  core_->GraphicsCore()->CreateBuffer(sizeof(uint32_t) + sizeof(float) * light_metadatas_.size(),
                                      graphics::BUFFER_TYPE_STATIC, &light_selector_buffer_);
  core_->GraphicsCore()->CreateBuffer(sizeof(LightMetadata) * light_metadatas_.size(), graphics::BUFFER_TYPE_STATIC,
                                      &light_metadatas_buffer_);
  light_metadatas_buffer_->UploadData(light_metadatas_.data(), sizeof(LightMetadata) * light_metadatas_.size());
  uint32_t light_count = static_cast<uint32_t>(light_metadatas_.size());
  light_selector_buffer_->UploadData(&light_count, sizeof(uint32_t), 0);

  core_->GraphicsCore()->CreateComputeProgram(gather_light_power_shader_.get(), &gather_light_power_program_);
  gather_light_power_program_->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, 1);
  gather_light_power_program_->AddResourceBinding(graphics::RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER, 1);
  gather_light_power_program_->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, buffers_.size());
  gather_light_power_program_->Finalize();

  rt_program_->AddResourceBinding(graphics::RESOURCE_TYPE_IMAGE, 1);
  rt_program_->AddResourceBinding(graphics::RESOURCE_TYPE_IMAGE, 1);
  rt_program_->AddResourceBinding(graphics::RESOURCE_TYPE_ACCELERATION_STRUCTURE, 1);
  rt_program_->AddResourceBinding(graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);
  rt_program_->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, 1);
  rt_program_->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, buffers_.size());
  rt_program_->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, 1);
  rt_program_->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, 1);
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

  preprocess_cmd_context_->CmdBindComputeProgram(gather_light_power_program_.get());
  preprocess_cmd_context_->CmdBindResources(0, {light_metadatas_buffer_.get()}, graphics::BIND_POINT_COMPUTE);
  preprocess_cmd_context_->CmdBindResources(1, {light_selector_buffer_.get()}, graphics::BIND_POINT_COMPUTE);
  preprocess_cmd_context_->CmdBindResources(2, buffers_, graphics::BIND_POINT_COMPUTE);
  preprocess_cmd_context_->CmdDispatch((light_metadatas_.size() + 63) / 64, 1, 1);

  auto blelloch_scan_up_program = core_->GetComputeProgram("blelloch_scan_up");
  auto blelloch_scan_down_program = core_->GetComputeProgram("blelloch_scan_down");
  blelloch_metadatas_.clear();
  BlellochScanMetadata metadata{4, 4, static_cast<uint32_t>(light_metadatas_.size())};
  uint32_t wave_size = core_->GraphicsCore()->WaveSize();
  while (metadata.element_count > 1) {
    blelloch_metadatas_.push_back(metadata);
    metadata = {metadata.offset + (wave_size - 1) * metadata.stride, metadata.stride * wave_size,
                metadata.element_count / wave_size};
  }
  if (!blelloch_metadatas_.empty()) {
    core_->GraphicsCore()->CreateBuffer(sizeof(BlellochScanMetadata) * blelloch_metadatas_.size(),
                                        graphics::BUFFER_TYPE_STATIC, &blelloch_metadata_buffer_);
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
  // std::vector<float> light_power_cdf(light_metadatas_.size(), 0.0f);
  // light_selector_buffer_->DownloadData(light_power_cdf.data(), light_power_cdf.size() * sizeof(float),
  //                                      sizeof(uint32_t));
  // for (size_t i = 0; i < light_power_cdf.size(); i++) {
  //   std::cout << "Light " << i << " Power CDF: " << light_power_cdf[i] << std::endl;
  // }
}

}  // namespace sparks
