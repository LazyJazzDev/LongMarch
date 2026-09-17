#include "sparkium/backends/offline_scene.h"

#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>
#include <cstring>
#include <map>
#include <stdexcept>

#include "grassland/util/file_probe.h"
#include "sparkium/backends/bvh_build.h"
#include "sparkium/backends/core/geometry.h"
#include "sparkium/sparkium.h"

namespace sparkium::backends {

namespace {

using device::float2;
using device::float3;
using device::float4;

float3 ToFloat3(const glm::vec3 &value) {
  return float3(value.x, value.y, value.z);
}

Mat4x3 ToMat4x3(const glm::mat4x3 &matrix) {
  Mat4x3 result;
  result.c0 = ToFloat3(matrix[0]);
  result.c1 = ToFloat3(matrix[1]);
  result.c2 = ToFloat3(matrix[2]);
  result.c3 = ToFloat3(matrix[3]);
  return result;
}

// LoadFloat4x4 transposes the uploaded column-major glm matrix, so row i of the
// ported matrix is row i of `matrix`.
device::float4x4 ToMat4(const glm::mat4 &matrix) {
  auto row = [&](int i) {
    return float4(matrix[0][i], matrix[1][i], matrix[2][i], matrix[3][i]);
  };
  return device::float4x4(row(0), row(1), row(2), row(3));
}

float MaxComponent(const glm::vec3 &value) {
  return std::max(value.x, std::max(value.y, value.z));
}

std::vector<uint8_t> DownloadBuffer(graphics::Buffer *buffer) {
  std::vector<uint8_t> data(buffer->Size());
  if (!data.empty())
    buffer->DownloadData(data.data(), data.size());
  return data;
}

bool ImageIsSupported(graphics::Image *image) {
  if (!image)
    return false;
  auto format = image->Format();
  return format == graphics::IMAGE_FORMAT_R8G8B8A8_UNORM ||
         format == graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT;
}

// Area of one mesh triangle under the light's own transform; MeshSamplePrimitive
// is reused so the geometry decoding stays identical to the shading core.
float PrimitiveArea(const DeviceScene &scene, uint32_t mesh_index, const Mat4x3 &transform, uint32_t primitive) {
  GeometryPrimitiveSample sample =
      MeshSamplePrimitive(MeshBuffer(scene, mesh_index), transform, primitive, float2(0.5f, 0.5f));
  return sample.pdf > 0.0f ? 1.0f / sample.pdf : 0.0f;
}

// Mirrors the per-material MaterialEvaluator::PrimitivePower of
// shaders/material/*/evaluator.hlsli, including the light power formula
// MaterialLight uses for its two-sided flag.
float MaterialPrimitivePower(const MaterialData &material, float area) {
  switch (material.kind) {
    case MATERIAL_KIND_LIGHT: {
      float emission = std::max(material.emission.x, std::max(material.emission.y, material.emission.z));
      float result = emission * area * SPARKIUM_PI;
      if (material.two_sided)
        result *= 2.0f;
      return result;
    }
    case MATERIAL_KIND_LAMBERTIAN: {
      float emission = std::max(material.emission.x, std::max(material.emission.y, material.emission.z));
      return emission * area * SPARKIUM_PI * 2.0f;
    }
    case MATERIAL_KIND_PRINCIPLED: {
      float3 emission = material.principled.emission_color;
      float max_emission = std::max(emission.x, std::max(emission.y, emission.z));
      return max_emission * material.principled.emission_strength * area * SPARKIUM_PI * 2.0f;
    }
    default:
      return 0.0f;
  }
}

int32_t MeshLightSamplerIndex(const MaterialData &material) {
  switch (material.kind) {
    case MATERIAL_KIND_LIGHT:
      return LIGHT_SAMPLER_MESH_LIGHT;
    case MATERIAL_KIND_LAMBERTIAN:
      return LIGHT_SAMPLER_MESH_LAMBERTIAN;
    case MATERIAL_KIND_PRINCIPLED:
      return LIGHT_SAMPLER_MESH_PRINCIPLED;
    case MATERIAL_KIND_SPECULAR:
      return LIGHT_SAMPLER_MESH_SPECULAR;
    default:
      return LIGHT_SAMPLER_MESH_SHADER_GRAPH;
  }
}

uint64_t MixBits(uint64_t hash, uint64_t value) {
  for (int byte = 0; byte < 8; ++byte) {
    hash ^= (value >> (8 * byte)) & 0xffull;
    hash *= 1099511628211ull;
  }
  return hash;
}

uint64_t MixFloat(uint64_t hash, float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return MixBits(hash, bits);
}

}  // namespace

std::unique_ptr<OfflineScene> OfflineScene::Build(sparkium::Scene *scene, sparkium::Camera *camera) {
  if (!scene || !camera)
    throw std::runtime_error("offline backends require a scene and a camera");
  auto result = std::unique_ptr<OfflineScene>(new OfflineScene());

  std::map<sparkium::Material *, uint32_t> material_indices;
  std::map<sparkium::Geometry *, uint32_t> mesh_indices;
  std::map<graphics::Image *, int32_t> texture_indices;

  auto register_material = [&](sparkium::Material *material) -> uint32_t {
    auto found = material_indices.find(material);
    if (found != material_indices.end())
      return found->second;
    MaterialData data{};
    data.kind = MATERIAL_KIND_LAMBERTIAN;
    data.base_color = float3(0.8f, 0.8f, 0.8f);
    data.emission = float3(0.0f, 0.0f, 0.0f);
    data.principled = PrincipledParams{};
    for (int i = 0; i < PRINCIPLED_TEXTURE_SLOT_COUNT; ++i)
      data.principled.texture_index[i] = -1;
    data.principled.normal_y_signal = 1.0f;
    data.principled.ior = 1.45f;
    data.principled.roughness = 0.5f;
    data.principled.subsurface_color = float3(1.0f, 1.0f, 1.0f);
    data.principled.subsurface_radius = float3(1.0f, 0.2f, 0.1f);
    data.principled.emission_color = float3(1.0f, 1.0f, 1.0f);

    if (auto *lambertian = dynamic_cast<MaterialLambertian *>(material)) {
      data.kind = MATERIAL_KIND_LAMBERTIAN;
      data.base_color = ToFloat3(lambertian->base_color);
      data.emission = ToFloat3(lambertian->emission);
    } else if (auto *light = dynamic_cast<MaterialLight *>(material)) {
      data.kind = MATERIAL_KIND_LIGHT;
      data.emission = ToFloat3(light->emission);
      data.two_sided = light->two_sided;
      data.block_ray = light->block_ray;
      data.camera_visible = light->camera_visible;
      data.falloff_distance = light->falloff_distance;
    } else if (auto *specular = dynamic_cast<MaterialSpecular *>(material)) {
      data.kind = MATERIAL_KIND_SPECULAR;
      data.base_color = ToFloat3(specular->base_color);
    } else if (auto *principled = dynamic_cast<MaterialPrincipled *>(material)) {
      data.kind = MATERIAL_KIND_PRINCIPLED;
      const MaterialPrincipled::Info &info = principled->info;
      PrincipledParams &params = data.principled;
      params.base_color = ToFloat3(info.base_color);
      params.subsurface_color = ToFloat3(info.subsurface_color);
      params.subsurface = info.subsurface;
      params.subsurface_radius = ToFloat3(info.subsurface_radius);
      params.metallic = info.metallic;
      params.specular = info.specular;
      params.specular_tint = info.specular_tint;
      params.roughness = info.roughness;
      params.anisotropic = info.anisotropic;
      params.anisotropic_rotation = info.anisotropic_rotation;
      params.sheen = info.sheen;
      params.sheen_tint = info.sheen_tint;
      params.clearcoat = info.clearcoat;
      params.clearcoat_roughness = info.clearcoat_roughness;
      params.ior = info.ior;
      params.transmission = info.transmission;
      params.transmission_roughness = info.transmission_roughness;
      params.emission_color = ToFloat3(info.emission_color);
      params.emission_strength = info.emission_strength;

      auto register_texture = [&](graphics::Image *image) -> int32_t {
        if (!ImageIsSupported(image))
          return -1;
        auto found_texture = texture_indices.find(image);
        if (found_texture != texture_indices.end())
          return found_texture->second;
        const bool hdr = image->Format() == graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT;
        graphics::Extent2D extent = image->Extent();
        TextureData texture{};
        texture.offset = static_cast<uint32_t>(result->texture_pixels_.size());
        texture.width = extent.width;
        texture.height = extent.height;
        texture.hdr = hdr ? 1 : 0;
        std::vector<float> pixels(static_cast<size_t>(extent.width) * extent.height * 4);
        if (hdr) {
          image->DownloadData(pixels.data());
        } else {
          std::vector<uint8_t> bytes(static_cast<size_t>(extent.width) * extent.height * 4);
          image->DownloadData(bytes.data());
          for (size_t i = 0; i < bytes.size(); ++i)
            pixels[i] = static_cast<float>(bytes[i]) / 255.0f;
        }
        result->texture_pixels_.insert(result->texture_pixels_.end(), pixels.begin(), pixels.end());
        int32_t index = hdr ? (static_cast<int32_t>(result->textures_.size()) | 0x1000000)
                            : static_cast<int32_t>(result->textures_.size());
        result->textures_.push_back(texture);
        texture_indices[image] = index;
        return index;
      };

      params.texture_index[PRINCIPLED_TEXTURE_BASE_COLOR] = register_texture(principled->textures.base_color);
      params.texture_index[PRINCIPLED_TEXTURE_EMISSION] = register_texture(principled->textures.emission);
      // Register the normal map first so the sign choice is explicit even when
      // the image is missing.
      params.texture_index[PRINCIPLED_TEXTURE_NORMAL] = register_texture(principled->textures.normal);
      params.normal_y_signal = principled->textures.normal_reverse_y ? -1.0f : 1.0f;
      params.texture_index[PRINCIPLED_TEXTURE_METALLIC] = register_texture(principled->textures.metallic);
      params.texture_index[PRINCIPLED_TEXTURE_SPECULAR] = register_texture(principled->textures.specular);
      params.texture_index[PRINCIPLED_TEXTURE_ROUGHNESS] = register_texture(principled->textures.roughness);
      params.texture_index[PRINCIPLED_TEXTURE_ANISOTROPIC] = register_texture(principled->textures.anisotropic);
      params.texture_index[PRINCIPLED_TEXTURE_ANISOTROPIC_ROTATION] =
          register_texture(principled->textures.anisotropic_rotation);
    } else if (dynamic_cast<MaterialShaderGraph *>(material)) {
      throw std::runtime_error(
          "offline backends do not support shader-graph materials (their surface shader is generated from the "
          "scene graph and has no portable counterpart)");
    } else {
      throw std::runtime_error("offline backends do not support this material type");
    }

    uint32_t index = static_cast<uint32_t>(result->materials_.size());
    result->materials_.push_back(data);
    material_indices[material] = index;
    return index;
  };

  uint32_t next_instance = 0;
  uint32_t next_light = 0;
  for (sparkium::Entity *entity : scene->GetEntityOrder()) {
    auto status = scene->GetEntities().find(entity);
    if (status == scene->GetEntities().end() || !status->second.active)
      continue;
    if (auto *geometry_material = dynamic_cast<sparkium::EntityGeometryMaterial *>(entity)) {
      sparkium::Geometry *geometry = geometry_material->GetGeometry();
      sparkium::Material *material = geometry_material->GetMaterial();
      if (!geometry || !material || geometry->PrimitiveCount() == 0)
        continue;  // mirrors EntityGeometryMaterial::Update
      auto *mesh = dynamic_cast<sparkium::GeometryMesh *>(geometry);
      if (!mesh)
        throw std::runtime_error("offline backends only support triangle mesh geometry");
      const glm::mat4x3 &transform = geometry_material->GetTransformation();
      glm::mat4 object_to_world(transform);
      if (std::abs(glm::determinant(object_to_world)) < 1.0e-20f)
        throw std::runtime_error("offline ray tracing requires invertible instance transforms");
      glm::mat4 world_to_object = glm::inverse(object_to_world);

      auto mesh_found = mesh_indices.find(geometry);
      if (mesh_found == mesh_indices.end()) {
        std::vector<uint8_t> blob = DownloadBuffer(mesh->GetBuffer());
        // Keep every mesh 16-byte aligned so the CUDA mirrors can use vector loads.
        while (result->mesh_data_.size() % 16 != 0)
          result->mesh_data_.push_back(0);
        MeshRange range{};
        range.offset = static_cast<uint32_t>(result->mesh_data_.size());
        range.primitive_count = static_cast<uint32_t>(geometry->PrimitiveCount());
        result->mesh_data_.insert(result->mesh_data_.end(), blob.begin(), blob.end());
        uint32_t root = BuildMeshTree(result->mesh_nodes_, result->mesh_data_.data() + range.offset,
                                     range.primitive_count);
        range.root = root;
        uint32_t index = static_cast<uint32_t>(result->meshes_.size());
        result->meshes_.push_back(range);
        mesh_indices[geometry] = index;
        // The light power pass below samples primitive areas through the
        // ported core, which reads the flat mesh data.
        result->RefreshDevicePointers();
      }

      uint32_t material_index = register_material(material);
      uint32_t mesh_index = mesh_indices[geometry];
      uint32_t instance_index = next_instance++;
      uint32_t light_index = next_light++;

      LightData light{};
      light.sampler_shader_index = MeshLightSamplerIndex(result->materials_[material_index]);
      light.custom_index = static_cast<int32_t>(instance_index);
      light.mesh_transform = ToMat4x3(transform);
      light.primitive_count = result->meshes_[mesh_index].primitive_count;
      light.primitive_cdf_offset = static_cast<uint32_t>(result->primitive_power_cdf_.size());
      float running = 0.0f;
      for (uint32_t primitive = 0; primitive < light.primitive_count; ++primitive) {
        float area = PrimitiveArea(result->device_, mesh_index, light.mesh_transform, primitive);
        running += MaterialPrimitivePower(result->materials_[material_index], area);
        result->primitive_power_cdf_.push_back(running);
      }
      light.power = running;
      result->lights_.push_back(light);

      InstanceData instance{};
      instance.object_to_world = ToMat4x3(transform);
      instance.world_to_object = ToMat4x3(glm::mat4x3(world_to_object));
      instance.mesh = mesh_index;
      instance.material = material_index;
      instance.light = static_cast<int32_t>(light_index);
      result->instances_.push_back(instance);
    } else if (auto *point_light = dynamic_cast<sparkium::EntityPointLight *>(entity)) {
      LightData light{};
      light.sampler_shader_index = LIGHT_SAMPLER_POINT;
      light.custom_index = -1;
      glm::vec3 power = point_light->color * point_light->strength;
      // LightPoint::SamplerPreprocess stores the selection weight at data[6].
      light.power = point_light->sampling_weight >= 0.0f ? point_light->sampling_weight : MaxComponent(power);
      light.light_position = ToFloat3(point_light->position);
      light.light_power = ToFloat3(power);
      light.radius = std::max(point_light->radius, 0.0f);
      light.soft_falloff = point_light->soft_falloff;
      result->lights_.push_back(light);
      ++next_light;
    }
  }

  // Inclusive prefix sums: for <= 64 lights the online gather kernel performs a
  // single-group inclusive scan, which is exactly this prefix sum.
  float running_power = 0.0f;
  for (const LightData &light : result->lights_) {
    running_power += light.power;
    result->light_power_cdf_.push_back(running_power);
  }

  // The instance tree bounds each mesh by its local root node, so every mesh
  // tree has to exist first.
  if (!result->instances_.empty())
    BuildInstanceTree(result->instance_nodes_, result->mesh_nodes_.data(), result->instances_.data(),
                      static_cast<uint32_t>(result->instances_.size()), result->meshes_.data());

  // Camera (raytracing::CameraData, resolved from the live camera).
  const float half_fov = std::tan(camera->fovy * 0.5f);
  result->camera_scale_[0] = camera->aspect * half_fov;
  result->camera_scale_[1] = half_fov;
  result->aperture_[0] = camera->aperture_radius;
  result->aperture_[1] = camera->focus_distance;
  result->aperture_[2] = static_cast<float>(camera->aperture_blades);
  result->aperture_[3] = camera->aperture_rotation;
  result->aperture_[4] = camera->aperture_ratio;
  glm::mat4 camera_to_world = glm::inverse(camera->view);
  for (int row = 0; row < 4; ++row)
    for (int column = 0; column < 4; ++column)
      result->camera_to_world_[row * 4 + column] = camera_to_world[column][row];

  result->EnsureSobolRows(1);
  return result;
}

bool OfflineScene::EnsureSobolRows(uint32_t rows) {
  if (rows > kSobolMaxRows)
    rows = kSobolMaxRows;  // RandomUint falls back to WangHash beyond the table
  if (rows == 0)
    rows = 1;
  if (SobolRowCount() >= rows && !sobol_rows_.empty())
    return false;
  sobol_rows_ = GenerateSobolRows(rows, FindAssetFile("data/new-joe-kuo-7.21201"));
  RefreshDevicePointers();
  return true;
}

void OfflineScene::RefreshDevicePointers() {
  device_.mesh_data = mesh_data_.data();
  device_.meshes = meshes_.data();
  device_.num_meshes = static_cast<uint32_t>(meshes_.size());
  device_.mesh_nodes = mesh_nodes_.data();
  device_.instance_nodes = instance_nodes_.data();
  device_.instances = instances_.data();
  device_.num_instances = static_cast<uint32_t>(instances_.size());
  device_.materials = materials_.data();
  device_.num_materials = static_cast<uint32_t>(materials_.size());
  device_.lights = lights_.data();
  device_.num_lights = static_cast<uint32_t>(lights_.size());
  device_.light_power_cdf = light_power_cdf_.data();
  device_.primitive_power_cdf = primitive_power_cdf_.data();
  device_.textures = textures_.data();
  device_.num_textures = static_cast<uint32_t>(textures_.size());
  device_.texture_pixels = texture_pixels_.data();
  device_.sobol_table = sobol_rows_.empty() ? nullptr : sobol_rows_.data();
  device_.camera_scale = float2(camera_scale_[0], camera_scale_[1]);
  device_.aperture_radius = aperture_[0];
  device_.focus_distance = aperture_[1];
  device_.aperture_blades = static_cast<int32_t>(aperture_[2]);
  device_.aperture_rotation = aperture_[3];
  device_.aperture_ratio = aperture_[4];
  device_.camera_to_world =
      device::float4x4(float4(camera_to_world_[0], camera_to_world_[1], camera_to_world_[2], camera_to_world_[3]),
                       float4(camera_to_world_[4], camera_to_world_[5], camera_to_world_[6], camera_to_world_[7]),
                       float4(camera_to_world_[8], camera_to_world_[9], camera_to_world_[10], camera_to_world_[11]),
                       float4(camera_to_world_[12], camera_to_world_[13], camera_to_world_[14],
                              camera_to_world_[15]));
}

std::string OfflineScene::Description() const {
  std::string result = std::to_string(instances_.size()) + " instance(s), " + std::to_string(materials_.size()) +
                       " material(s), " + std::to_string(lights_.size()) + " light(s), " +
                       std::to_string(textures_.size()) + " texture(s), " +
                       std::to_string(mesh_nodes_.size() + instance_nodes_.size()) + " BVH node(s), " +
                       std::to_string(mesh_data_.size()) + " mesh byte(s)";
  return result;
}

std::vector<uint64_t> OfflineSceneSignature(sparkium::Scene *scene, sparkium::Camera *camera) {
  std::vector<uint64_t> signature;
  if (!scene || !camera)
    return signature;
  uint64_t hash = 1469598103934665603ull;
  uint64_t entity_count = 0;
  for (sparkium::Entity *entity : scene->GetEntityOrder()) {
    auto status = scene->GetEntities().find(entity);
    bool active = status != scene->GetEntities().end() && status->second.active;
    hash = MixBits(hash, reinterpret_cast<uint64_t>(entity));
    hash = MixBits(hash, active ? 1u : 0u);
    if (auto *geometry_material = dynamic_cast<sparkium::EntityGeometryMaterial *>(entity)) {
      hash = MixBits(hash, reinterpret_cast<uint64_t>(geometry_material->GetGeometry()));
      hash = MixBits(hash, reinterpret_cast<uint64_t>(geometry_material->GetMaterial()));
      const glm::mat4x3 &transform = geometry_material->GetTransformation();
      for (int column = 0; column < 4; ++column)
        for (int row = 0; row < 3; ++row)
          hash = MixFloat(hash, transform[column][row]);
    } else if (auto *point_light = dynamic_cast<sparkium::EntityPointLight *>(entity)) {
      hash = MixFloat(hash, point_light->position.x);
      hash = MixFloat(hash, point_light->position.y);
      hash = MixFloat(hash, point_light->position.z);
      hash = MixFloat(hash, point_light->color.x);
      hash = MixFloat(hash, point_light->color.y);
      hash = MixFloat(hash, point_light->color.z);
      hash = MixFloat(hash, point_light->strength);
      hash = MixFloat(hash, point_light->radius);
      hash = MixFloat(hash, point_light->sampling_weight);
      hash = MixBits(hash, static_cast<uint64_t>(point_light->soft_falloff));
    }
    ++entity_count;
  }
  hash = MixBits(hash, scene->settings.samples_per_dispatch);
  hash = MixBits(hash, scene->settings.max_bounces);
  hash = MixBits(hash, scene->settings.alpha_shadow);
  hash = MixFloat(hash, scene->settings.background_color.x);
  hash = MixFloat(hash, scene->settings.background_color.y);
  hash = MixFloat(hash, scene->settings.background_color.z);
  for (int row = 0; row < 4; ++row)
    for (int column = 0; column < 4; ++column)
      hash = MixFloat(hash, camera->view[column][row]);
  hash = MixFloat(hash, camera->fovy);
  hash = MixFloat(hash, camera->aspect);
  hash = MixFloat(hash, camera->aperture_radius);
  hash = MixFloat(hash, camera->focus_distance);
  hash = MixBits(hash, camera->aperture_blades);
  hash = MixFloat(hash, camera->aperture_rotation);
  hash = MixFloat(hash, camera->aperture_ratio);
  signature.push_back(entity_count);
  signature.push_back(hash);
  return signature;
}

}  // namespace sparkium::backends
