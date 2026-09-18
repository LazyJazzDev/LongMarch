#include "sparkium/pipelines/native/core/scene_data.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "grassland/util/file_probe.h"
#include "grassland/util/sobol.h"
#include "sparkium/core/core.h"

namespace sparkium::native {

namespace {

ByteBuffer MakeBuffer(const std::vector<uint32_t> &data) {
  ByteBuffer buffer;
  buffer.data = data.data();
  buffer.size = static_cast<uint32_t>(data.size() * sizeof(uint32_t));
  return buffer;
}

// Appends raw bytes to a word buffer, padding the tail with zeros the way a
// four-byte aligned `graphics::Buffer` allocation does.
void AppendBytes(std::vector<uint32_t> &words, const void *data, size_t size) {
  const size_t base = words.size();
  words.resize(base + (size + 3) / 4, 0);
  std::memcpy(words.data() + base, data, size);
}

void AppendFloat3(std::vector<uint32_t> &words, const glm::vec3 &value) {
  AppendBytes(words, &value, sizeof(value));
}

float MaxChannel(const glm::vec3 &value) {
  return std::max(std::max(value.x, value.y), value.z);
}

// `geometry/mesh/geometry_sampler.hlsli::PrimitiveArea`.
float PrimitiveArea(const ByteBuffer &geometry, const float3x4 &transform, uint32_t primitive_id) {
  const uint32_t position_offset = geometry.Load(8);
  const uint32_t position_stride = geometry.Load(12);
  const uint32_t index_offset = geometry.Load(48);
  const uint3 vid = geometry.Load3(index_offset + primitive_id * 3 * 4);
  float3 pos[3];
  for (int i = 0; i < 3; ++i)
    pos[i] = mul(transform, float4{LoadFloat3(geometry, position_offset + position_stride * vid[i]), 1.0f});
  return glm::length(glm::cross(pos[1] - pos[0], pos[2] - pos[0])) * 0.5f;
}

// The five `material/*/evaluator.hlsli::PrimitivePower` implementations.
float PrimitivePower(uint32_t material_type, const ByteBuffer &material, float area) {
  switch (material_type) {
    case NATIVE_MATERIAL_LIGHT: {
      float power = MaxChannel(LoadFloat3(material, 0)) * area * PI;
      if (material.Load(12))
        power *= 2.0f;
      return power;
    }
    case NATIVE_MATERIAL_LAMBERTIAN:
      return MaxChannel(LoadFloat3(material, 12)) * area * PI * 2.0f;
    case NATIVE_MATERIAL_PRINCIPLED: {
      const float4 emission = LoadFloat4(material, 92);
      return std::max(std::max(emission.x, emission.y), emission.z) * emission.w * area * PI * 2.0f;
    }
    case NATIVE_MATERIAL_SHADER_GRAPH:
      return MaxChannel(LoadFloat3(material, 0)) * area * PI * 2.0f;
    default:  // Specular emits nothing.
      return 0.0f;
  }
}

constexpr uint32_t kGroupSize = 64;

// Group-local inclusive prefix sum over 64 lanes, as left behind by the
// `WavePrefixSum` plus groupshared reduction that every gather kernel and
// `BlellochUpSweep` share. Reproducing the grouping rather than calling
// `std::partial_sum` keeps the floating-point summation order identical.
void GroupPrefixSum(std::vector<uint32_t> &words, uint32_t offset, uint32_t stride, uint32_t count) {
  float *base = reinterpret_cast<float *>(words.data());
  for (uint32_t group_first = 0; group_first < count; group_first += kGroupSize) {
    float running = 0.0f;
    const uint32_t group_last = std::min(group_first + kGroupSize, count);
    for (uint32_t index = group_first; index < group_last; ++index) {
      float &element = base[(offset + index * stride) / 4];
      running += element;
      element = running;
    }
  }
}

// `blelloch_scan.hlsl::BlellochDownSweep`. The lanes it writes and the lanes it
// reads are disjoint, so a serial loop matches the parallel dispatch exactly.
void DownSweep(std::vector<uint32_t> &words, uint32_t offset, uint32_t stride, uint32_t count) {
  float *base = reinterpret_cast<float *>(words.data());
  for (uint32_t index = kGroupSize; index < count; ++index) {
    if (index % kGroupSize == kGroupSize - 1)
      continue;
    const uint32_t add_index = (index / kGroupSize - 1) * kGroupSize + kGroupSize - 1;
    base[(offset + index * stride) / 4] += base[(offset + add_index * stride) / 4];
  }
}

// The scan chain built by `Scene::UpdatePipeline` and
// `LightGeometryMaterial::SamplerPreprocess`, which differ only in the
// down-sweep skip predicate.
void HierarchicalPrefixSum(std::vector<uint32_t> &words,
                           uint32_t offset,
                           uint32_t stride,
                           uint32_t count,
                           bool skip_equal_group) {
  struct Level {
    uint32_t offset, stride, count;
  };
  std::vector<Level> levels;
  Level level{offset, stride, count};
  while (level.count > 1) {
    levels.push_back(level);
    level = {level.offset + (kGroupSize - 1) * level.stride, level.stride * kGroupSize, level.count / kGroupSize};
  }
  // Level 0 was already reduced by the power-gathering kernel.
  for (size_t i = 1; i < levels.size(); ++i)
    GroupPrefixSum(words, levels[i].offset, levels[i].stride, levels[i].count);
  for (size_t i = levels.size(); i-- > 0;) {
    if (skip_equal_group ? levels[i].count <= kGroupSize : levels[i].count < kGroupSize)
      continue;
    DownSweep(words, levels[i].offset, levels[i].stride, levels[i].count);
  }
}

}  // namespace

SceneData::SceneData(sparkium::Core *core) : core_(core) {
  // The same table `Core::LoadPublicBuffers` uploads as the "sobol" buffer.
  sobol_ = SobolTableGen(65536, 1024, FindAssetFile("data/new-joe-kuo-7.21201"));
}

int32_t SceneData::RegisterBuffer(const void *key, std::vector<uint32_t> &&data) {
  if (!key)
    return -1;
  auto existing = buffer_index_.find(key);
  if (existing != buffer_index_.end())
    return existing->second;
  auto &cached = blob_cache_[key];
  if (cached.data != data) {
    cached.data = std::move(data);
    cached.revision = next_revision_++;
  }
  const int32_t index = static_cast<int32_t>(data_buffers_.size());
  data_buffers_.push_back({key, cached.revision, &cached.data});
  buffer_index_[key] = index;
  return index;
}

int32_t SceneData::RegisterImage(graphics::Image *image) {
  if (!image)
    return -1;
  const bool sdr = image->Format() == graphics::IMAGE_FORMAT_R8G8B8A8_UNORM;
  const bool hdr = image->Format() == graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT;
  if (!sdr && !hdr)
    return -1;

  auto &index_map = sdr ? sdr_index_ : hdr_index_;
  auto existing = index_map.find(image);
  if (existing != index_map.end())
    return existing->second + (hdr ? 0x1000000 : 0);

  auto cached = texture_cache_.find(image);
  if (cached == texture_cache_.end()) {
    // Texture pixels only live in graphics memory. Copying them out is an asset
    // transfer, not part of the rendering computation.
    const auto extent = image->Extent();
    HostTexture texture;
    texture.width = static_cast<int>(extent.width);
    texture.height = static_cast<int>(extent.height);
    texture.revision = next_revision_++;
    const size_t pixels = static_cast<size_t>(texture.width) * static_cast<size_t>(texture.height);
    if (sdr) {
      texture.sdr.resize(pixels);
      image->DownloadData(texture.sdr.data());
    } else {
      texture.hdr.resize(pixels * 4);
      image->DownloadData(texture.hdr.data());
    }
    cached = texture_cache_.emplace(image, std::move(texture)).first;
  }

  auto &list = sdr ? sdr_textures_ : hdr_textures_;
  const int32_t index = static_cast<int32_t>(list.size());
  list.push_back(&cached->second);
  index_map[image] = index;
  return index + (hdr ? 0x1000000 : 0);
}

int32_t SceneData::RegisterMaterial(sparkium::Material *material) {
  auto existing = material_index_.find(material);
  if (existing != material_index_.end())
    return existing->second;

  NativeMaterial native{};
  native.graph_index = -1;
  if (dynamic_cast<sparkium::MaterialLambertian *>(material)) {
    native.type = NATIVE_MATERIAL_LAMBERTIAN;
  } else if (dynamic_cast<sparkium::MaterialSpecular *>(material)) {
    native.type = NATIVE_MATERIAL_SPECULAR;
  } else if (dynamic_cast<sparkium::MaterialLight *>(material)) {
    native.type = NATIVE_MATERIAL_LIGHT;
  } else if (dynamic_cast<sparkium::MaterialPrincipled *>(material)) {
    native.type = NATIVE_MATERIAL_PRINCIPLED;
  } else if (auto *graph = dynamic_cast<sparkium::MaterialShaderGraph *>(material)) {
    native.type = NATIVE_MATERIAL_SHADER_GRAPH;
    native.graph_index = static_cast<int32_t>(graph_programs_.size());
    const auto &program = graph->program;
    if (program.instructions.empty() && static_cast<bool>(graph->graph_code))
      throw std::runtime_error(
          "the native backends need a compiled shader-graph program, but this material carries HLSL only");
    HostGraphProgram host{};
    host.instructions.reserve(program.instructions.size());
    for (const auto &instruction : program.instructions) {
      GraphInstruction device{};
      device.op = instruction.op;
      device.sub = instruction.sub;
      device.dst = instruction.dst;
      for (int i = 0; i < GRAPH_MAX_OPERANDS; ++i)
        device.operands[i] = instruction.operands[i];
      device.data_offset = instruction.data_offset;
      device.data_count = instruction.data_count;
      host.instructions.push_back(device);
    }
    host.constants = program.constants;
    host.data = program.data;
    for (int i = 0; i < GRAPH_SURFACE_OUTPUT_COUNT; ++i)
      host.outputs[i] = program.outputs[i];
    graph_programs_.push_back(std::move(host));
  } else {
    throw std::runtime_error("the native backends do not support this material type");
  }

  const int32_t index = static_cast<int32_t>(materials_.size());
  materials_.push_back(native);
  material_index_[material] = index;
  return index;
}

std::vector<uint32_t> SceneData::GeometryBlob(sparkium::GeometryMesh *geometry) {
  auto *buffer = geometry->GetBuffer();
  std::vector<uint32_t> words((buffer->Size() + 3) / 4, 0);
  buffer->DownloadData(words.data(), buffer->Size());
  return words;
}

std::vector<uint32_t> SceneData::MaterialBlob(sparkium::Material *material) {
  std::vector<uint32_t> words;
  if (auto *lambertian = dynamic_cast<sparkium::MaterialLambertian *>(material)) {
    AppendFloat3(words, lambertian->base_color);
    AppendFloat3(words, lambertian->emission);
  } else if (auto *specular = dynamic_cast<sparkium::MaterialSpecular *>(material)) {
    AppendFloat3(words, specular->base_color);
  } else if (auto *light = dynamic_cast<sparkium::MaterialLight *>(material)) {
    AppendFloat3(words, light->emission);
    words.push_back(static_cast<uint32_t>(light->two_sided));
    words.push_back(static_cast<uint32_t>(light->block_ray));
    words.push_back(static_cast<uint32_t>(light->camera_visible));
    words.push_back(asuint(light->falloff_distance));
  } else if (auto *principled = dynamic_cast<sparkium::MaterialPrincipled *>(material)) {
    AppendBytes(words, &principled->info, sizeof(principled->info));
    // `RegisteredTextures`, uploaded right behind `Info`.
    const auto &textures = principled->textures;
    words.push_back(static_cast<uint32_t>(textures.normal ? RegisterImage(textures.normal) : -1));
    words.push_back(asuint(textures.normal && textures.normal_reverse_y ? -1.0f : 1.0f));
    for (auto *image : {textures.base_color, textures.metallic, textures.specular, textures.roughness,
                        textures.anisotropic, textures.anisotropic_rotation, textures.emission})
      words.push_back(static_cast<uint32_t>(image ? RegisterImage(image) : -1));
  } else if (auto *graph = dynamic_cast<sparkium::MaterialShaderGraph *>(material)) {
    AppendFloat3(words, graph->emission_hint);
    for (auto *image : graph->textures)
      words.push_back(static_cast<uint32_t>(RegisterImage(image)));
    words.resize(std::max<size_t>(4, words.size()), 0);  // max(16, 12 + textures * 4) bytes
  } else {
    throw std::runtime_error("the native backends do not support this material type");
  }
  return words;
}

int32_t SceneData::MeshLightSamplerShader(sparkium::Material *material) const {
  if (dynamic_cast<sparkium::MaterialLight *>(material))
    return 0x1000001;
  if (dynamic_cast<sparkium::MaterialLambertian *>(material))
    return 0x1000002;
  if (dynamic_cast<sparkium::MaterialPrincipled *>(material))
    return 0x1000003;
  if (dynamic_cast<sparkium::MaterialShaderGraph *>(material))
    return 0x1000004;
  if (dynamic_cast<sparkium::MaterialSpecular *>(material))
    return 0x1000005;
  throw std::runtime_error("the native backends have no light sampler for this material type");
}

std::vector<uint32_t> SceneData::MeshLightBlob(uint32_t primitive_count,
                                               uint32_t material_type,
                                               const std::vector<uint32_t> &geometry_blob,
                                               const std::vector<uint32_t> &material_blob,
                                               const glm::mat4x3 &transform) {
  std::vector<uint32_t> words;
  AppendBytes(words, &transform, sizeof(transform));  // float3x4 transform, offset 0
  words.push_back(primitive_count);                   // uint primitive_count, offset 48
  words.resize(13 + primitive_count, 0);              // float cdf[primitive_count], offset 52

  const ByteBuffer geometry_buffer = MakeBuffer(geometry_blob);
  const ByteBuffer material_buffer = MakeBuffer(material_blob);
  float *cdf = reinterpret_cast<float *>(words.data()) + 13;
  for (uint32_t primitive = 0; primitive < primitive_count; ++primitive)
    cdf[primitive] =
        PrimitivePower(material_type, material_buffer, PrimitiveArea(geometry_buffer, transform, primitive));
  // `gather_primitive_power.hlsl` leaves the group-local prefix sums behind and
  // the Blelloch chain completes the scan.
  GroupPrefixSum(words, 52, 4, primitive_count);
  HierarchicalPrefixSum(words, 52, 4, primitive_count, false);
  return words;
}

void SceneData::UpdateGeometryMaterial(sparkium::EntityGeometryMaterial *entity) {
  auto *geometry = dynamic_cast<sparkium::GeometryMesh *>(entity->GetGeometry());
  auto *material = entity->GetMaterial();
  if (!geometry || !material)
    throw std::runtime_error("the native backends currently require triangle mesh geometry");
  if (geometry->PrimitiveCount() == 0)
    return;

  const glm::mat4x3 transform = entity->GetTransformation();
  const uint32_t primitive_count = static_cast<uint32_t>(geometry->PrimitiveCount());

  // Texture registration happens first, exactly as `Material::Update(scene)`.
  auto material_blob = MaterialBlob(material);
  const int32_t material_index = RegisterMaterial(material);
  auto geometry_blob = GeometryBlob(geometry);

  // `Scene::RegisterLight`: sampler data, then shader index and power offset.
  const int32_t light_index = static_cast<int32_t>(light_metadatas_.size() / 4);
  const int32_t sampler_data_index = RegisterBuffer(
      entity,
      MeshLightBlob(primitive_count, materials_[material_index].type, geometry_blob, material_blob, transform));
  light_metadatas_.push_back(static_cast<uint32_t>(MeshLightSamplerShader(material)));
  light_metadatas_.push_back(static_cast<uint32_t>(sampler_data_index));
  light_metadatas_.push_back(0);  // custom_index, patched below.
  light_metadatas_.push_back(52 + (primitive_count - 1) * 4);

  // `Scene::RegisterSoftwareInstance`: geometry buffer, then material buffer.
  const int32_t instance_index = static_cast<int32_t>(instance_metadatas_.size() / 3);
  const int32_t geometry_index = RegisterBuffer(geometry->GetBuffer(), std::move(geometry_blob));
  const int32_t material_data_index = RegisterBuffer(material, std::move(material_blob));
  instance_metadatas_.push_back(static_cast<uint32_t>(geometry_index));
  instance_metadatas_.push_back(static_cast<uint32_t>(material_data_index));
  instance_metadatas_.push_back(static_cast<uint32_t>(light_index));

  const glm::mat4 object_to_world(transform);
  if (std::abs(glm::determinant(object_to_world)) < 1.0e-20f)
    throw std::runtime_error("software ray tracing requires invertible instance transforms");
  BvhInstance instance{};
  instance.object_to_world = transform;
  instance.world_to_object = glm::mat4x3(glm::inverse(object_to_world));
  instance.root = 0;  // Assigned by BuildInstanceRecords().
  instance.geometry = static_cast<uint32_t>(geometry_index);
  instance.material = static_cast<uint32_t>(material_index);
  instance.primitive_count = primitive_count;
  instances_.push_back(instance);
  instance_geometries_.push_back(geometry);

  light_metadatas_[light_index * 4 + 2] = static_cast<uint32_t>(instance_index);
}

void SceneData::UpdatePointLight(sparkium::EntityPointLight *entity) {
  // `LightPoint::SamplerPreprocess` writes nine floats.
  const glm::vec3 power = entity->color * entity->strength;
  std::vector<uint32_t> words;
  AppendFloat3(words, entity->position);
  AppendFloat3(words, power);
  words.push_back(asuint(entity->sampling_weight >= 0.0f ? entity->sampling_weight : MaxChannel(power)));
  words.push_back(asuint(std::max(entity->radius, 0.0f)));
  words.push_back(static_cast<uint32_t>(entity->soft_falloff));

  const int32_t sampler_data_index = RegisterBuffer(entity, std::move(words));
  light_metadatas_.push_back(0x1000000);
  light_metadatas_.push_back(static_cast<uint32_t>(sampler_data_index));
  light_metadatas_.push_back(static_cast<uint32_t>(-1));
  light_metadatas_.push_back(24);
}

void SceneData::BuildLightSelector(uint32_t light_count) {
  light_selector_.assign(1 + light_count, 0);
  light_selector_[0] = light_count;
  float *cdf = reinterpret_cast<float *>(light_selector_.data()) + 1;
  for (uint32_t i = 0; i < light_count; ++i) {
    const int32_t data_index = static_cast<int32_t>(light_metadatas_[i * 4 + 1]);
    const uint32_t power_offset = light_metadatas_[i * 4 + 3];
    cdf[i] = asfloat(MakeBuffer(*data_buffers_[data_index].data).Load(power_offset));
  }
  // `gather_light_power.hlsl` plus the Blelloch chain from `UpdatePipeline`.
  GroupPrefixSum(light_selector_, 4, 4, light_count);
  HierarchicalPrefixSum(light_selector_, 4, 4, light_count, true);
}

void SceneData::BuildInstanceRecords() {
  // Node budget and per-geometry roots, following `SoftwarePipeline::Update`.
  const uint32_t tlas_leaves = LeafCount(instances_.size());
  uint64_t node_count = uint64_t(tlas_leaves) * 2 - 1;
  std::vector<BvhGeometry> geometries;
  std::vector<sparkium::GeometryMesh *> geometry_keys;
  for (size_t i = 0; i < instances_.size(); ++i) {
    auto *mesh = instance_geometries_[i];
    auto found = std::find(geometry_keys.begin(), geometry_keys.end(), mesh);
    if (found == geometry_keys.end()) {
      const uint32_t count = static_cast<uint32_t>(mesh->PrimitiveCount());
      const uint32_t leaves = LeafCount(count);
      if (node_count + uint64_t(leaves) * 2 - 1 > 0xffffffffull / 32u)
        throw std::runtime_error("software BVH node address overflow");
      geometries.push_back(
          {data_buffers_[instances_[i].geometry].data, static_cast<uint32_t>(node_count), leaves, count});
      geometry_keys.push_back(mesh);
      node_count += uint64_t(leaves) * 2 - 1;
      found = geometry_keys.end() - 1;
    }
    instances_[i].root = geometries[std::distance(geometry_keys.begin(), found)].root;
  }

  // 16-byte header followed by the 112-byte instance records.
  instance_records_.assign(4, 0);
  instance_records_[0] = static_cast<uint32_t>(instances_.size());
  for (const auto &instance : instances_) {
    AppendBytes(instance_records_, &instance.object_to_world, sizeof(instance.object_to_world));
    AppendBytes(instance_records_, &instance.world_to_object, sizeof(instance.world_to_object));
    instance_records_.push_back(instance.root);
    instance_records_.push_back(instance.geometry);
    instance_records_.push_back(instance.material);
    instance_records_.push_back(instance.primitive_count);
  }

  bvh_.Build(geometries, instances_, tlas_leaves, static_cast<uint32_t>(node_count));
  geometry_revision_ = next_revision_++;
}

void SceneData::BuildView(const sparkium::Scene::Settings::RayTracing &settings,
                          const sparkium::Film::Info &film_info) {
  view_data_buffers_.clear();
  for (const auto &buffer : data_buffers_)
    view_data_buffers_.push_back(MakeBuffer(*buffer.data));
  view_sdr_textures_.clear();
  for (const auto *texture : sdr_textures_)
    view_sdr_textures_.push_back(
        {texture->sdr.empty() ? nullptr : texture->sdr.data(), nullptr, texture->width, texture->height});
  view_hdr_textures_.clear();
  for (const auto *texture : hdr_textures_)
    view_hdr_textures_.push_back(
        {nullptr, texture->hdr.empty() ? nullptr : texture->hdr.data(), texture->width, texture->height});
  view_graph_programs_.clear();
  for (const auto &program : graph_programs_) {
    GraphProgram device{};
    device.instructions = program.instructions.data();
    device.instruction_count = static_cast<uint32_t>(program.instructions.size());
    device.constants = program.constants.data();
    device.data = program.data.data();
    for (int i = 0; i < GRAPH_SURFACE_OUTPUT_COUNT; ++i)
      device.outputs[i] = program.outputs[i];
    view_graph_programs_.push_back(device);
  }

  view_ = SceneView{};
  view_.software_nodes = MakeBuffer(bvh_.Nodes());
  view_.software_instances = MakeBuffer(instance_records_);
  view_.data_buffers = view_data_buffers_.data();
  view_.data_buffer_count = static_cast<uint32_t>(view_data_buffers_.size());
  view_.sobol_table = MakeBuffer(sobol_);
  view_.camera_data = MakeBuffer(camera_data_);
  view_.instance_metadatas = MakeBuffer(instance_metadatas_);
  view_.light_selector_data = MakeBuffer(light_selector_);
  view_.light_metadatas = MakeBuffer(light_metadatas_);
  view_.sdr_textures = view_sdr_textures_.data();
  view_.sdr_texture_count = static_cast<uint32_t>(view_sdr_textures_.size());
  view_.hdr_textures = view_hdr_textures_.data();
  view_.hdr_texture_count = static_cast<uint32_t>(view_hdr_textures_.size());
  view_.materials = materials_.data();
  view_.material_count = static_cast<uint32_t>(materials_.size());
  view_.graph_programs = view_graph_programs_.data();
  view_.graph_program_count = static_cast<uint32_t>(view_graph_programs_.size());

  // `scene_settings_buffer_`: Settings::RayTracing followed by Film::Info.
  view_.settings.samples_per_dispatch = settings.samples_per_dispatch;
  view_.settings.max_bounces = settings.max_bounces;
  view_.settings.alpha_shadow = settings.alpha_shadow;
  view_.settings._settings_padding = 0;
  view_.settings.background_color = settings.background_color;
  view_.settings.accumulated_samples = film_info.accumulated_samples;
  view_.settings.persistence = film_info.persistence;
  view_.settings.clamping = film_info.clamping;
  view_.settings.max_exposure = film_info.max_exposure;
  view_.settings.view_transform = film_info.view_transform;
  view_.settings.exposure = film_info.exposure;
  view_.settings.gamma = film_info.gamma;
  view_.settings.contrast = film_info.contrast;
}

void SceneData::Update(sparkium::Scene *scene, sparkium::Camera *camera, const sparkium::Film::Info &film_info) {
  data_buffers_.clear();
  buffer_index_.clear();
  sdr_textures_.clear();
  sdr_index_.clear();
  hdr_textures_.clear();
  hdr_index_.clear();
  materials_.clear();
  material_index_.clear();
  graph_programs_.clear();
  instance_metadatas_.clear();
  light_metadatas_.clear();
  instances_.clear();
  instance_geometries_.clear();

  // `UpdatePipeline` registers the two default images before walking the
  // entities, so texture index 0 is always `white`.
  RegisterImage(core_->GetImage("white"));
  RegisterImage(core_->GetImage("white_hdr"));

  // `raytracing::CameraData`.
  camera_data_.clear();
  {
    const glm::mat4 world_to_camera = camera->view;
    const glm::mat4 camera_to_world = glm::inverse(camera->view);
    const glm::vec2 scale{camera->aspect * std::tan(camera->fovy * 0.5f), std::tan(camera->fovy * 0.5f)};
    AppendBytes(camera_data_, &world_to_camera, sizeof(world_to_camera));
    AppendBytes(camera_data_, &camera_to_world, sizeof(camera_to_world));
    AppendBytes(camera_data_, &scale, sizeof(scale));
    camera_data_.push_back(asuint(camera->aperture_radius));
    camera_data_.push_back(asuint(camera->focus_distance));
    camera_data_.push_back(static_cast<uint32_t>(camera->aperture_blades));
    camera_data_.push_back(asuint(camera->aperture_rotation));
    camera_data_.push_back(asuint(camera->aperture_ratio));
  }

  const auto &entity_status = scene->GetEntities();
  for (auto *entity : scene->GetEntityOrder()) {
    if (!entity_status.at(entity).active)
      continue;
    if (auto *geometry_material = dynamic_cast<sparkium::EntityGeometryMaterial *>(entity))
      UpdateGeometryMaterial(geometry_material);
    else if (auto *point_light = dynamic_cast<sparkium::EntityPointLight *>(entity))
      UpdatePointLight(point_light);
    else
      throw std::runtime_error("the native backends do not support this entity type");
  }

  BuildLightSelector(static_cast<uint32_t>(light_metadatas_.size() / 4));
  BuildInstanceRecords();

  // `UpdatePipeline` registers the camera buffer when no entity registered one,
  // keeping `data_buffers[0]` valid for empty scenes.
  if (data_buffers_.empty())
    RegisterBuffer(camera, std::vector<uint32_t>(camera_data_));

  BuildView(scene->settings.raytracing, film_info);
}

}  // namespace sparkium::native
