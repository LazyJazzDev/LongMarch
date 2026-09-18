#include "sparkium/pipelines/portable/scene_bake.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <tuple>

#include "glm/gtc/constants.hpp"

#include "sparkium/core/camera.h"
#include "sparkium/core/film.h"
#include "sparkium/core/scene.h"
#include "sparkium/entity/entity_geometry_material.h"
#include "sparkium/entity/entity_point_light.h"
#include "sparkium/geometry/geometry_mesh.h"
#include "sparkium/material/material_light.h"
#include "sparkium/material/material_lambertian.h"
#include "sparkium/material/material_principled.h"
#include "sparkium/material/material_shader_graph.h"
#include "sparkium/material/material_specular.h"
#include "sparkium/pipelines/raytracing/core/software_pipeline.h"
#include "sparkium/pipelines/raytracing/geometry/geometries.h"
#include "sparkium/pipelines/raytracing/material/materials.h"

namespace sparkium::portable {
namespace {

constexpr uint32_t kInvalid = 0xffffffffu;

uint32_t LeafCount(size_t count) {
  if (count > (uint32_t(1) << 21))
    throw std::runtime_error("portable BVH exceeds 2097152 leaves per tree");
  uint32_t result = 1;
  while (result < count)
    result *= 2;
  return result;
}

struct Bounds {
  float lo[3]{FLT_MAX, FLT_MAX, FLT_MAX};
  float hi[3]{-FLT_MAX, -FLT_MAX, -FLT_MAX};
  bool Valid() const {
    return lo[0] <= hi[0] && lo[1] <= hi[1] && lo[2] <= hi[2];
  }
  void Extend(const float p[3]) {
    for (int i = 0; i < 3; ++i) {
      lo[i] = std::min(lo[i], p[i]);
      hi[i] = std::max(hi[i], p[i]);
    }
  }
  void Extend(const Bounds &b) {
    if (!b.Valid())
      return;
    Extend(b.lo);
    Extend(b.hi);
  }
};

// Loads the serialized GeometryMesh header (mirrors GeometryHeader in HLSL).
struct MeshHeader {
  uint32_t num_vertices, num_indices;
  uint32_t position_offset, position_stride;
  uint32_t normal_offset, normal_stride;
  uint32_t tex_coord_offset, tex_coord_stride;
  uint32_t tangent_offset, tangent_stride;
  uint32_t signal_offset, signal_stride;
  uint32_t index_offset;
  uint32_t color_offset, color_stride;
};

MeshHeader ReadHeader(const std::vector<uint8_t> &data) {
  MeshHeader h{};
  std::memcpy(&h, data.data(), sizeof(h));
  return h;
}

void TriangleBounds(const std::vector<uint8_t> &data, const MeshHeader &h, uint32_t primitive, float lo[3],
                    float hi[3]) {
  uint32_t ids[3];
  std::memcpy(ids, data.data() + h.index_offset + primitive * 12, 12);
  for (int i = 0; i < 3; ++i) {
    lo[i] = FLT_MAX;
    hi[i] = -FLT_MAX;
  }
  for (uint32_t v : ids) {
    float p[3];
    std::memcpy(p, data.data() + h.position_offset + h.position_stride * v, 12);
    for (int i = 0; i < 3; ++i) {
      lo[i] = std::min(lo[i], p[i]);
      hi[i] = std::max(hi[i], p[i]);
    }
  }
}

float TriangleAreaTransformed(const std::vector<uint8_t> &data, const MeshHeader &h, uint32_t primitive,
                              const glm::mat4x3 &t) {
  uint32_t ids[3];
  std::memcpy(ids, data.data() + h.index_offset + primitive * 12, 12);
  glm::vec3 pos[3];
  for (int v = 0; v < 3; ++v) {
    float p[3];
    std::memcpy(p, data.data() + h.position_offset + h.position_stride * ids[v], 12);
    pos[v] = t * glm::vec4(p[0], p[1], p[2], 1.0f);
  }
  return glm::length(glm::cross(pos[1] - pos[0], pos[2] - pos[0])) * 0.5f;
}

// Builds one complete binary tree (leaves rounded to a power of two) with the
// same Morton order as shaders/software/build.hlsl. Returns the node list;
// empty slots keep invalid bounds and are skipped by traversal.
std::vector<Node> BuildTree(uint32_t leaves,
                            uint32_t count,
                            const std::function<void(uint32_t primitive, Bounds &bounds)> &leaf_bounds) {
  std::vector<Node> nodes(size_t(leaves) * 2 - 1);
  std::vector<Bounds> bounds(leaves);
  for (auto &node : nodes) {
    node.first = kInvalid;
    node.second = kInvalid;
    for (int i = 0; i < 3; ++i) {
      node.lo[i] = FLT_MAX;
      node.hi[i] = -FLT_MAX;
    }
  }
  Bounds total;
  for (uint32_t i = 0; i < count; ++i) {
    leaf_bounds(i, bounds[i]);
    total.Extend(bounds[i]);
  }
  auto morton = [&](uint32_t i) {
    if (i >= count || !total.Valid() || !bounds[i].Valid())
      return kInvalid;
    float center[3], extent[3];
    uint32_t code = 0;
    for (int axis = 0; axis < 3; ++axis) {
      center[axis] = (bounds[i].lo[axis] + bounds[i].hi[axis]) * 0.5f;
      extent[axis] = std::max(total.hi[axis] - total.lo[axis], 1.0e-20f);
      float normalized = (center[axis] - total.lo[axis]) / extent[axis];
      normalized = std::min(std::max(normalized, 0.0f), 1.0f);
      uint32_t p = uint32_t(normalized * 1023.0f);
      // SpreadBits from build.hlsl.
      p = (p | (p << 16)) & 0x030000ffu;
      p = (p | (p << 8)) & 0x0300f00fu;
      p = (p | (p << 4)) & 0x030c30c3u;
      p = (p | (p << 2)) & 0x09249249u;
      code |= p << axis;
    }
    return code;
  };
  std::vector<uint32_t> order(leaves);
  for (uint32_t i = 0; i < leaves; ++i)
    order[i] = i;
  std::stable_sort(order.begin(), order.end(),
                   [&](uint32_t a, uint32_t b) { return std::make_pair(morton(a), a) < std::make_pair(morton(b), b); });
  for (uint32_t slot = 0; slot < leaves; ++slot) {
    Node &node = nodes[leaves - 1 + slot];
    const uint32_t primitive = order[slot];
    if (primitive < count && bounds[primitive].Valid()) {
      node.second = primitive;
      for (int i = 0; i < 3; ++i) {
        node.lo[i] = bounds[primitive].lo[i];
        node.hi[i] = bounds[primitive].hi[i];
      }
    }
  }
  for (uint32_t count_level = leaves / 2; count_level; count_level /= 2) {
    const uint32_t first = count_level - 1;
    for (uint32_t i = 0; i < count_level; ++i) {
      const uint32_t index = first + i;
      Node &node = nodes[index];
      node.first = index * 2 + 1;
      node.second = node.first + 1;
      Bounds b;
      const Node &a = nodes[node.first];
      const Node &c = nodes[node.second];
      b.Extend(Bounds{a.lo[0], a.lo[1], a.lo[2], a.hi[0], a.hi[1], a.hi[2]});
      b.Extend(Bounds{c.lo[0], c.lo[1], c.lo[2], c.hi[0], c.hi[1], c.hi[2]});
      for (int k = 0; k < 3; ++k) {
        node.lo[k] = b.lo[k];
        node.hi[k] = b.hi[k];
      }
    }
  }
  return nodes;
}


struct Baker {
  sparkium::Scene *scene;
  sparkium::Camera *camera;
  sparkium::Film *film;
  const std::map<graphics::Image *, const HostImageData *> &host_images;
  uint32_t *seed;

  BakeResult result;
  std::map<sparkium::Geometry *, uint32_t> geometry_index;
  // Deduplicated shader variants (codegen), mirroring the compute renderer's
  // material switch.
  std::map<std::pair<bool, std::string>, uint32_t> shader_index;
  // Per-material-object buffers (mirrors Scene::RegisterBuffer(material->Buffer())).
  std::map<sparkium::Material *, uint32_t> material_buffer_index;
  std::map<graphics::Image *, int32_t> sdr_index;
  std::map<graphics::Image *, int32_t> hdr_index;
  std::mt19937 rng;

  int32_t RegisterImage(graphics::Image *image);
  uint32_t RegisterGeometry(sparkium::GeometryMesh *geometry);
  uint32_t RegisterMaterial(sparkium::Material *material);
  float EmissionHint(sparkium::Material *material);
  void AddMeshInstance(sparkium::GeometryMesh *geometry,
                       sparkium::Material *material,
                       const glm::mat4x3 &transform,
                       uint32_t instance_custom_index);
  void AddGeometryMaterialEntity(sparkium::EntityGeometryMaterial *entity);
  void AddPointLight(sparkium::EntityPointLight *entity);
  void FinalizeLights();
};

void WriteMat4x3(float *dst, const glm::mat4x3 &m) {
  // Match the reference Vulkan software pipeline, which uploads the
  // glm::mat4x3 by value: glm is column-major, so the 12 floats are four
  // columns of three floats. The shared HLSL LoadFloat3x4 reads exactly this
  // column-major layout (four float3 columns). Writing row-major here would
  // transpose every instance transform and break all world_to_object ray
  // transforms.
  for (int col = 0; col < 4; ++col)
    for (int row = 0; row < 3; ++row)
      dst[col * 3 + row] = m[col][row];
}

int32_t Baker::RegisterImage(graphics::Image *image) {
  const auto host = host_images.find(image);
  if (host == host_images.end())
    throw std::runtime_error("portable backend: texture without host data (image not loaded through JsonScene)");
  const HostImageData *data = host->second;
  if (data->format == graphics::IMAGE_FORMAT_R8G8B8A8_UNORM) {
    if (!sdr_index.count(image)) {
      sdr_index[image] = static_cast<int32_t>(result.sdr_textures.size());
      TextureData entry;
      entry.pixels = data->pixels;
      entry.texture.data = entry.pixels.data();
      entry.texture.width = uint32_t(data->width);
      entry.texture.height = uint32_t(data->height);
      entry.texture.format = sparkium_portable::TEXTURE_FORMAT_SDR;
      result.sdr_textures.push_back(std::move(entry));
    }
    return sdr_index[image];
  }
  if (!hdr_index.count(image)) {
    hdr_index[image] = static_cast<int32_t>(result.hdr_textures.size());
    TextureData entry;
    entry.pixels = data->pixels;
    entry.texture.data = entry.pixels.data();
    entry.texture.width = uint32_t(data->width);
    entry.texture.height = uint32_t(data->height);
    entry.texture.format = sparkium_portable::TEXTURE_FORMAT_HDR;
    result.hdr_textures.push_back(std::move(entry));
  }
  return hdr_index[image] + 0x1000000;
}

uint32_t Baker::RegisterGeometry(sparkium::GeometryMesh *geometry) {
  if (geometry_index.count(geometry))
    return geometry_index[geometry];
  const uint32_t index = static_cast<uint32_t>(result.geometries.size());
  geometry_index[geometry] = index;
  GeometryData entry;
  entry.buffer = geometry->GetData();
  entry.count = uint32_t(geometry->PrimitiveCount());
  entry.leaves = LeafCount(entry.count);
  entry.buffer_index = index;
  const MeshHeader h = ReadHeader(entry.buffer);
  auto nodes = BuildTree(entry.leaves, entry.count, [&](uint32_t primitive, Bounds &bounds) {
    float lo[3], hi[3];
    TriangleBounds(entry.buffer, h, primitive, lo, hi);
    bounds = Bounds{lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]};
  });
  entry.root = 0;  // patched when nodes are appended
  result.geometries.push_back(std::move(entry));
  // Nodes are appended later in instance order; store the tree temporarily.
  // We append immediately to keep indexing simple.
  GeometryData &stored = result.geometries.back();
  stored.root = static_cast<uint32_t>(result.nodes.size());
  result.nodes.insert(result.nodes.end(), std::make_move_iterator(nodes.begin()), std::make_move_iterator(nodes.end()));
  return index;
}

uint32_t Baker::RegisterMaterial(sparkium::Material *material) {
  auto *rt_material = raytracing::DedicatedCast(material);
  const bool is_graph = rt_material->GraphImpl() != nullptr;
  const std::string source = raytracing::SoftwarePipeline::SharedMaterialSource(
      is_graph ? *rt_material->GraphImpl() : rt_material->SamplerImpl());
  const auto key = std::make_pair(is_graph, source);
  // Distinct material objects may share the same shader source (e.g. two
  // lambertian materials with different colors). Keep the codegen variant
  // deduplicated, but serialize one data buffer per material object, exactly
  // like Scene::RegisterBuffer(material->Buffer()) in the raytracing
  // pipeline. Returns the material-buffer index in the unified data-buffer
  // table ([geometries..., material buffers..., lights...]).
  if (material_buffer_index.count(material))
    return material_buffer_index[material];

  if (!shader_index.count(key)) {
    const uint32_t variant = static_cast<uint32_t>(result.materials.size());
    shader_index[key] = variant;
    MaterialData entry;
    entry.shader_graph = is_graph;
    entry.shadow_any_hit = source.find("#define SAMPLE_SHADOW_ANY_HIT") != std::string::npos;
    entry.shadow_no_hitrecord = source.find("#define SAMPLE_SHADOW_NO_HITRECORD") != std::string::npos;
    entry.source = source;
    result.materials.push_back(std::move(entry));
  }
  MaterialBufferData entry;
  entry.shader_index = shader_index[key];

  // Serialize the material buffer exactly like the raytracing pipeline.
  if (auto *lambertian = dynamic_cast<MaterialLambertian *>(material)) {
    entry.buffer.resize(sizeof(glm::vec3) * 2);
    std::memcpy(entry.buffer.data(), &lambertian->base_color, sizeof(glm::vec3));
    std::memcpy(entry.buffer.data() + sizeof(glm::vec3), &lambertian->emission, sizeof(glm::vec3));
  } else if (auto *light = dynamic_cast<MaterialLight *>(material)) {
    entry.buffer.resize(28);
    std::memcpy(entry.buffer.data(), &light->emission, sizeof(glm::vec3));
    std::memcpy(entry.buffer.data() + 12, &light->two_sided, sizeof(int));
    std::memcpy(entry.buffer.data() + 16, &light->block_ray, sizeof(int));
    std::memcpy(entry.buffer.data() + 20, &light->camera_visible, sizeof(int));
    std::memcpy(entry.buffer.data() + 24, &light->falloff_distance, sizeof(float));
  } else if (auto *specular = dynamic_cast<MaterialSpecular *>(material)) {
    entry.buffer.resize(sizeof(glm::vec3));
    std::memcpy(entry.buffer.data(), &specular->base_color, sizeof(glm::vec3));
  } else if (auto *principled = dynamic_cast<MaterialPrincipled *>(material)) {
    struct RegisteredTextures {
      int normal{-1};
      float y_signal{1.0f};
      int base_color{-1};
      int metallic{-1};
      int specular{-1};
      int roughness{-1};
      int anisotropic{-1};
      int anisotropic_rotation{-1};
      int emission{-1};
    } registered{};
    auto &textures = principled->textures;
    if (textures.normal) {
      registered.normal = RegisterImage(textures.normal);
      registered.y_signal = textures.normal_reverse_y ? -1.0f : 1.0f;
    }
    if (textures.base_color)
      registered.base_color = RegisterImage(textures.base_color);
    if (textures.metallic)
      registered.metallic = RegisterImage(textures.metallic);
    if (textures.specular)
      registered.specular = RegisterImage(textures.specular);
    if (textures.roughness)
      registered.roughness = RegisterImage(textures.roughness);
    if (textures.anisotropic)
      registered.anisotropic = RegisterImage(textures.anisotropic);
    if (textures.anisotropic_rotation)
      registered.anisotropic_rotation = RegisterImage(textures.anisotropic_rotation);
    if (textures.emission)
      registered.emission = RegisterImage(textures.emission);
    entry.buffer.resize(sizeof(principled->info) + sizeof(registered));
    std::memcpy(entry.buffer.data(), &principled->info, sizeof(principled->info));
    std::memcpy(entry.buffer.data() + sizeof(principled->info), &registered, sizeof(registered));
  } else if (auto *graph = dynamic_cast<MaterialShaderGraph *>(material)) {
    entry.buffer.resize(std::max<size_t>(16, 12 + graph->textures.size() * sizeof(int)), 0);
    std::memcpy(entry.buffer.data(), &graph->emission_hint, sizeof(glm::vec3));
    std::vector<int> indices;
    indices.reserve(graph->textures.size());
    for (auto *texture : graph->textures)
      indices.push_back(RegisterImage(texture));
    if (!indices.empty())
      std::memcpy(entry.buffer.data() + 12, indices.data(), indices.size() * sizeof(int));
  } else {
    throw std::runtime_error("portable backend: unsupported material type");
  }
  const uint32_t index = static_cast<uint32_t>(result.material_buffers.size());
  material_buffer_index[material] = index;
  result.material_buffers.push_back(std::move(entry));
  return index;
}

float Baker::EmissionHint(sparkium::Material *material) {
  glm::vec3 emission{0.0f};
  float scale = 1.0f;
  if (auto *lambertian = dynamic_cast<MaterialLambertian *>(material)) {
    emission = lambertian->emission;
    scale = 2.0f;
  } else if (auto *light = dynamic_cast<MaterialLight *>(material)) {
    emission = light->emission;
    scale = light->two_sided ? 2.0f : 1.0f;
  } else if (auto *principled = dynamic_cast<MaterialPrincipled *>(material)) {
    emission = principled->emission_color * principled->emission_strength;
    scale = 2.0f;
  } else if (auto *graph = dynamic_cast<MaterialShaderGraph *>(material)) {
    emission = graph->emission_hint;
    scale = 2.0f;
  } else if (dynamic_cast<MaterialSpecular *>(material)) {
    return 0.0f;
  }
  return std::max(std::max(emission.x, emission.y), emission.z) * scale;
}

void Baker::AddMeshInstance(sparkium::GeometryMesh *geometry,
                            sparkium::Material *material,
                            const glm::mat4x3 &transform,
                            uint32_t instance_index) {
  const uint32_t geometry_idx = RegisterGeometry(geometry);
  RegisterMaterial(material);
  // Instance::material is the deduplicated shader variant, mirroring the
  // compute renderer's SoftwarePipeline::Update material switch.
  const uint32_t material_idx = result.material_buffers[material_buffer_index[material]].shader_index;
  const GeometryData &geo = result.geometries[geometry_idx];
  Instance instance{};
  WriteMat4x3(instance.object_to_world, transform);
  const glm::mat4 o2w(transform);
  if (std::abs(glm::determinant(o2w)) < 1.0e-20f)
    throw std::runtime_error("portable backend requires invertible instance transforms");
  WriteMat4x3(instance.world_to_object, glm::mat4x3(glm::inverse(o2w)));
  instance.root = geo.root;
  instance.geometry = geometry_idx;
  instance.material = material_idx;
  instance.primitive_count = geo.count;
  result.instances.push_back(instance);

  // Build the light sampler CDF for emissive meshes.
  LightData light;
  const float hint = EmissionHint(material);
  light.buffer.resize(sizeof(glm::mat4x3) + sizeof(uint32_t) + geo.count * sizeof(float));
  std::memcpy(light.buffer.data(), &transform, sizeof(glm::mat4x3));
  std::memcpy(light.buffer.data() + sizeof(glm::mat4x3), &geo.count, sizeof(uint32_t));
  const MeshHeader h = ReadHeader(geo.buffer);
  float accumulated = 0.0f;
  for (uint32_t p = 0; p < geo.count; ++p) {
    const float area = TriangleAreaTransformed(geo.buffer, h, p, transform);
    accumulated += hint * area * glm::pi<float>();
    std::memcpy(light.buffer.data() + sizeof(glm::mat4x3) + sizeof(uint32_t) + p * sizeof(float), &accumulated,
                sizeof(float));
  }
  light.metadata.power_offset =
      sizeof(glm::mat4x3) + sizeof(uint32_t) + (geo.count ? (geo.count - 1) * sizeof(float) : 0);
  light.metadata.custom_index = int32_t(instance_index);
  result.lights.push_back(std::move(light));
}

void Baker::AddGeometryMaterialEntity(sparkium::EntityGeometryMaterial *entity) {
  auto *geometry = dynamic_cast<sparkium::GeometryMesh *>(entity->GetGeometry());
  auto *material = entity->GetMaterial();
  if (!geometry || !material || geometry->PrimitiveCount() == 0)
    return;
  const uint32_t instance_index = static_cast<uint32_t>(result.instances.size());
  const uint32_t light_index = static_cast<uint32_t>(result.lights.size());
  AddMeshInstance(geometry, material, entity->GetTransformation(), instance_index);

  GpuInstanceMetadata metadata{};
  metadata.geometry_data_index = int32_t(geometry_index[geometry]);
  // NOTE: stored as the material-buffer-local index here; FinalizeLights
  // rebases it into the unified data-buffer table ([geometries...,
  // material buffers..., lights...]) once the final geometry count is known.
  metadata.material_data_index = int32_t(material_buffer_index[material]);
  metadata.custom_index = int32_t(light_index);
  result.instance_metadatas.push_back(metadata);

  // Mesh light sampler IDs (mirrors LightGeometryMaterial::SamplerShader).
  LightData &light = result.lights.back();
  if (dynamic_cast<MaterialLight *>(material))
    light.metadata.sampler_shader_index = 0x1000001;
  else if (dynamic_cast<MaterialLambertian *>(material))
    light.metadata.sampler_shader_index = 0x1000002;
  else if (dynamic_cast<MaterialPrincipled *>(material))
    light.metadata.sampler_shader_index = 0x1000003;
  else if (dynamic_cast<MaterialShaderGraph *>(material))
    light.metadata.sampler_shader_index = 0x1000004;
  else if (dynamic_cast<MaterialSpecular *>(material))
    light.metadata.sampler_shader_index = 0x1000005;
  // sampler_data_index is assigned in FinalizeLights (data buffer table).
}

void Baker::AddPointLight(sparkium::EntityPointLight *entity) {
  LightData light;
  light.buffer.resize(sizeof(float) * 9);
  float data[9];
  const glm::vec3 power = entity->color * entity->strength;
  const float max_power = std::max(std::max(power.r, power.g), power.b);
  std::memcpy(data, &entity->position, sizeof(glm::vec3));
  std::memcpy(data + 3, &power, sizeof(glm::vec3));
  data[6] = entity->sampling_weight >= 0.0f ? entity->sampling_weight : max_power;
  data[7] = std::max(entity->radius, 0.0f);
  std::memcpy(data + 8, &entity->soft_falloff, sizeof(int));
  std::memcpy(light.buffer.data(), data, sizeof(data));
  light.metadata.sampler_shader_index = 0x1000000;
  light.metadata.custom_index = -1;
  light.metadata.power_offset = sizeof(glm::vec3) + sizeof(glm::vec3);
  result.lights.push_back(std::move(light));
}

void Baker::FinalizeLights() {
  // Data buffer table: [geometry buffers..., material buffers..., light
  // sampler buffers...]. sampler_data_index points into that table.
  const uint32_t geometry_count = static_cast<uint32_t>(result.geometries.size());
  const uint32_t material_count = static_cast<uint32_t>(result.material_buffers.size());
  for (uint32_t i = 0; i < result.lights.size(); ++i)
    result.lights[i].metadata.sampler_data_index = int32_t(geometry_count + material_count + i);
  // Rebase material_data_index from material-local to the unified data-buffer
  // table (materials follow all geometry buffers).
  for (auto &metadata : result.instance_metadatas)
    if (metadata.material_data_index >= 0)
      metadata.material_data_index += int32_t(geometry_count);
  // Light power CDF (prefix sum of per-light power at power_offset).
  result.light_power_cdf.resize(result.lights.size());
  float accumulated = 0.0f;
  for (size_t i = 0; i < result.lights.size(); ++i) {
    const LightData &light = result.lights[i];
    float power = 0.0f;
    if (light.metadata.power_offset + 4 <= light.buffer.size())
      std::memcpy(&power, light.buffer.data() + light.metadata.power_offset, 4);
    accumulated += power;
    result.light_power_cdf[i] = accumulated;
  }
}

}  // namespace

BakeResult BakeScene(sparkium::Scene *scene,
                     sparkium::Camera *camera,
                     sparkium::Film *film,
                     const std::map<graphics::Image *, const HostImageData *> &host_images,
                     uint32_t *seed) {
  Baker baker{scene, camera, film, host_images, seed};
  baker.rng.seed(seed ? *seed : 0u);

  for (auto *entity : scene->GetEntityOrder()) {
    const auto &status = scene->GetEntities().at(entity);
    if (!status.active)
      continue;
    if (auto *gm = dynamic_cast<sparkium::EntityGeometryMaterial *>(entity)) {
      baker.AddGeometryMaterialEntity(gm);
    } else if (auto *pl = dynamic_cast<sparkium::EntityPointLight *>(entity)) {
      baker.AddPointLight(pl);
    } else {
      throw std::runtime_error("portable backend: unsupported entity type");
    }
  }

  // TLAS over instance world bounds (instances were appended in order).
  const uint32_t tlas_leaves = LeafCount(std::max<size_t>(1, baker.result.instances.size()));
  auto tlas = BuildTree(tlas_leaves, static_cast<uint32_t>(baker.result.instances.size()),
                        [&](uint32_t index, Bounds &bounds) {
                          const Instance &instance = baker.result.instances[index];
                          const GeometryData &geo = baker.result.geometries[instance.geometry];
                          const Node &root = baker.result.nodes[geo.root];
                          if (root.lo[0] > root.hi[0])
                            return;  // empty geometry
                          const glm::mat4x3 &transform =
                              *reinterpret_cast<const glm::mat4x3 *>(instance.object_to_world);
                          for (uint32_t corner = 0; corner < 8; ++corner) {
                            const float p[3] = {(corner & 1) ? root.hi[0] : root.lo[0],
                                                (corner & 2) ? root.hi[1] : root.lo[1],
                                                (corner & 4) ? root.hi[2] : root.lo[2]};
                            const glm::vec3 world = transform * glm::vec4(p[0], p[1], p[2], 1.0f);
                            bounds.Extend(&world.x);
                          }
                        });
  // The traversal starts at node 0, so the TLAS occupies the first slots and
  // geometry trees follow; shift the recorded geometry roots accordingly.
  {
    const uint32_t tlas_count = static_cast<uint32_t>(tlas.size());
    // BuildTree emits child indices local to each tree (rooted at 0), but
    // the traversal walks one global node array whose first tlas_count slots
    // are the TLAS. Rebase every BLAS internal child link to its global slot
    // (tlas_count + local_root + local_child). Leaves keep primitive ids in
    // `second` and are identified by first == kInvalid, so they are skipped.
    // This mirrors the reference GPU builder, which writes nodes directly at
    // their global offsets (BuildParameters.root).
    for (auto &geometry : baker.result.geometries) {
      const uint32_t node_count = geometry.leaves * 2 - 1;
      for (uint32_t i = 0; i < node_count; ++i) {
        Node &node = baker.result.nodes[geometry.root + i];
        if (node.first != kInvalid) {
          node.first += tlas_count + geometry.root;
          node.second += tlas_count + geometry.root;
        }
      }
    }
    baker.result.nodes.insert(baker.result.nodes.begin(), std::make_move_iterator(tlas.begin()),
                              std::make_move_iterator(tlas.end()));
    for (auto &geometry : baker.result.geometries) {
      geometry.root += tlas_count;
    }
    for (auto &instance : baker.result.instances)
      instance.root = baker.result.geometries[instance.geometry].root;
  }

  baker.FinalizeLights();

  // CameraData (160 bytes, mirroring shaders/camera.hlsl loads).
  baker.result.camera_data.assign(160, 0);
  {
    float *data = reinterpret_cast<float *>(baker.result.camera_data.data());
    const glm::mat4 world_to_camera = camera->view;
    const glm::mat4 camera_to_world = glm::inverse(camera->view);
    for (int row = 0; row < 4; ++row)
      for (int col = 0; col < 4; ++col)
        data[row * 4 + col] = world_to_camera[col][row];
    for (int row = 0; row < 4; ++row)
      for (int col = 0; col < 4; ++col)
        data[16 + row * 4 + col] = camera_to_world[col][row];
    data[32] = camera->aspect * std::tan(camera->fovy * 0.5f);
    data[33] = std::tan(camera->fovy * 0.5f);
    data[34] = camera->aperture_radius;
    data[35] = camera->focus_distance;
    int32_t blades = camera->aperture_blades;
    std::memcpy(baker.result.camera_data.data() + 144, &blades, 4);
    data[37] = camera->aperture_rotation;
    data[38] = camera->aperture_ratio;
  }

  // RenderSettings (64 bytes, HLSL RenderSettings layout).
  baker.result.render_settings.assign(64, 0);
  {
    const auto &rt = scene->settings.raytracing;
    const auto &info = film->info;
    int32_t ints[4] = {rt.samples_per_dispatch, rt.max_bounces, rt.alpha_shadow, 0};
    std::memcpy(baker.result.render_settings.data(), ints, 16);
    std::memcpy(baker.result.render_settings.data() + 16, &rt.background_color, 12);
    // HLSL struct layout: accumulated_samples lands at offset 28 (float3 at
    // 16 is 12 bytes; the following int stays 4-byte aligned).
    int32_t head[1] = {info.accumulated_samples};
    std::memcpy(baker.result.render_settings.data() + 28, head, 4);
    float floats[8] = {info.persistence, info.clamping, info.max_exposure, float(info.view_transform),
                       info.exposure,    info.gamma,    info.contrast,     0.0f};
    std::memcpy(baker.result.render_settings.data() + 32, floats, 32);
  }
  if (seed)
    *seed = baker.rng();
  return std::move(baker.result);
}

void ToneMapPixels(const sparkium_portable::float4 *averaged_color,
                   int view_transform,
                   float exposure,
                   float gamma,
                   float contrast,
                   uint32_t width,
                   uint32_t height,
                   uint8_t *rgba) {
  auto linear_to_srgb = [](float c) {
    return c <= 0.0031308f ? c * 12.92f : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
  };
  const float exposure_scale = std::exp2(exposure);
  for (uint32_t i = 0; i < width * height; ++i) {
    sparkium_portable::float3 linear(averaged_color[i][0] * exposure_scale,
                                     averaged_color[i][1] * exposure_scale,
                                     averaged_color[i][2] * exposure_scale);
    for (int k = 0; k < 3; ++k)
      linear[k] = std::max(linear[k], 0.0f);
    sparkium_portable::float3 mapped;
    if (view_transform == 1) {
      for (int k = 0; k < 3; ++k)
        mapped[k] = std::min(std::max(linear_to_srgb(linear[k]), 0.0f), 1.0f);
    } else if (view_transform == 2) {
      for (int k = 0; k < 3; ++k) {
        const float c = linear[k];
        mapped[k] = (c * (2.51f * c + 0.03f)) / (c * (2.43f * c + 0.59f) + 0.14f);
        mapped[k] = std::min(std::max(mapped[k], 0.0f), 1.0f);
        mapped[k] = (mapped[k] - 0.18f) * contrast + 0.18f;
        mapped[k] = std::pow(std::max(mapped[k], 0.0f), 1.0f / std::max(gamma, 1.0e-4f));
        mapped[k] = std::min(std::max(mapped[k], 0.0f), 1.0f);
      }
    } else {
      const float max_channel = std::max(linear.x, std::max(linear.y, linear.z));
      linear = linear / std::max(1.0f, max_channel);
      for (int k = 0; k < 3; ++k)
        mapped[k] = linear_to_srgb(linear[k]);
    }
    for (int k = 0; k < 3; ++k)
      rgba[i * 4 + k] = uint8_t(std::min(std::max(mapped[k], 0.0f), 1.0f) * 255.0f + 0.5f);
    rgba[i * 4 + 3] = 255;
  }
}

}  // namespace sparkium::portable
