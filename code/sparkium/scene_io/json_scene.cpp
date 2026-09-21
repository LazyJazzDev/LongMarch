#include "sparkium/scene_io/json_scene.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "glm/gtc/matrix_transform.hpp"
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "rapidjson/istreamwrapper.h"
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace sparkium {
namespace {
using namespace grassland;
using Value = rapidjson::Value;

const Value &RequireObject(const Value &value) {
  if (!value.IsObject())
    throw std::runtime_error("expected a JSON object");
  return value;
}

const Value &RequireArray(const Value &value) {
  if (!value.IsArray())
    throw std::runtime_error("expected a JSON array");
  return value;
}

std::string ReadString(const Value &value) {
  if (!value.IsString())
    throw std::runtime_error("expected a string");
  return value.GetString();
}

int ReadInt(const Value &value) {
  if (!value.IsInt())
    throw std::runtime_error("expected a 32-bit integer");
  return value.GetInt();
}

float ReadNumber(const Value &value) {
  if (!value.IsNumber())
    throw std::runtime_error("expected a number");
  float result = value.GetFloat();
  if (!std::isfinite(result))
    throw std::runtime_error("expected a finite float");
  return result;
}

bool ReadBool(const Value &value) {
  if (!value.IsBool())
    throw std::runtime_error("expected a boolean");
  return value.GetBool();
}

const Value &Member(const Value &value, const char *name) {
  if (!value.IsObject() || !value.HasMember(name))
    throw std::runtime_error(std::string("missing field '") + name + "'");
  return value[name];
}

glm::vec3 Vec3(const Value &value, const glm::vec3 &fallback = {}) {
  if (value.IsNull())
    return fallback;
  if (!value.IsArray() || value.Size() != 3)
    throw std::runtime_error("expected an array of three numbers");
  return {ReadNumber(value[0]), ReadNumber(value[1]), ReadNumber(value[2])};
}

glm::vec3 Vec3Member(const Value &value, const char *name, const glm::vec3 &fallback = {}) {
  return RequireObject(value).HasMember(name) ? Vec3(value[name], fallback) : fallback;
}

float FloatMember(const Value &value, const char *name, float fallback) {
  return RequireObject(value).HasMember(name) ? ReadNumber(value[name]) : fallback;
}

bool BoolMember(const Value &value, const char *name, bool fallback) {
  return RequireObject(value).HasMember(name) ? ReadBool(value[name]) : fallback;
}

std::filesystem::path Resolve(const std::filesystem::path &document, const std::string &asset) {
  std::filesystem::path path(asset);
  if (path.is_relative())
    path = document.parent_path() / path;
  return std::filesystem::weakly_canonical(path);
}

glm::mat4 Transform(const Value &value) {
  glm::mat4 result{1.0f};
  RequireObject(value);
  if (value.HasMember("matrix")) {
    const auto &matrix = value["matrix"];
    if (!matrix.IsArray() || matrix.Size() != 16)
      throw std::runtime_error("transform.matrix must contain 16 numbers");
    for (rapidjson::SizeType i = 0; i < 16; ++i)
      result[i / 4][i % 4] = ReadNumber(matrix[i]);
    return result;
  }
  result = glm::translate(result, Vec3Member(value, "translation"));
  auto rotation = glm::radians(Vec3Member(value, "rotation_degrees"));
  result = glm::rotate(result, rotation.x, {1.0f, 0.0f, 0.0f});
  result = glm::rotate(result, rotation.y, {0.0f, 1.0f, 0.0f});
  result = glm::rotate(result, rotation.z, {0.0f, 0.0f, 1.0f});
  result = glm::scale(result, Vec3Member(value, "scale", {1.0f, 1.0f, 1.0f}));
  return result;
}

Mesh<> InlineMesh(const Value &value) {
  const auto &positions_json = RequireArray(Member(value, "positions"));
  const auto &indices_json = RequireArray(Member(value, "indices"));
  std::vector<Vector3<float>> positions;
  std::vector<Vector2<float>> tex_coords;
  std::vector<uint32_t> indices;
  for (const auto &item : positions_json.GetArray()) {
    auto p = Vec3(item);
    positions.push_back({p.x, p.y, p.z});
  }
  for (const auto &item : indices_json.GetArray()) {
    if (!item.IsUint() || item.GetUint() >= positions.size())
      throw std::runtime_error("mesh index must reference an existing vertex");
    indices.push_back(item.GetUint());
  }
  if (positions.empty() || indices.empty() || indices.size() % 3 != 0)
    throw std::runtime_error("inline mesh must contain vertices and complete triangles");
  if (value.HasMember("tex_coords")) {
    for (const auto &item : RequireArray(value["tex_coords"]).GetArray()) {
      if (!item.IsArray() || item.Size() != 2)
        throw std::runtime_error("texture coordinate must have two numbers");
      tex_coords.push_back({ReadNumber(item[0]), ReadNumber(item[1])});
    }
  }
  if (value.HasMember("tex_coords") && tex_coords.size() != positions.size())
    throw std::runtime_error("mesh texture coordinates must match vertex count");
  return Mesh<>(positions.size(), indices.size(), indices.data(), positions.data(), nullptr,
                tex_coords.empty() ? nullptr : tex_coords.data());
}

Mesh<> BinaryMesh(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream)
    throw std::runtime_error("cannot open binary mesh: " + path.string());
  const auto size = stream.tellg();
  stream.seekg(0);
  std::array<char, 8> magic{};
  uint32_t num_vertices = 0, num_indices = 0, flags = 0;
  stream.read(magic.data(), magic.size());
  stream.read(reinterpret_cast<char *>(&num_vertices), sizeof(num_vertices));
  stream.read(reinterpret_cast<char *>(&num_indices), sizeof(num_indices));
  stream.read(reinterpret_cast<char *>(&flags), sizeof(flags));
  if (!stream || std::memcmp(magic.data(), "SPKMESH1", 8) != 0)
    throw std::runtime_error("invalid Sparkium binary mesh: " + path.string());
  if (num_indices % 3 != 0 || (flags & ~7u))
    throw std::runtime_error("invalid Sparkium binary mesh header: " + path.string());
  const uint64_t expected =
      20ull + uint64_t(num_indices) * sizeof(uint32_t) +
      uint64_t(num_vertices) * (sizeof(float) * 3 + ((flags & 1u) ? sizeof(float) * 3 : 0) +
                                ((flags & 2u) ? sizeof(float) * 2 : 0) + ((flags & 4u) ? sizeof(float) * 3 : 0));
  if (uint64_t(size) != expected)
    throw std::runtime_error("truncated or oversized Sparkium binary mesh: " + path.string());
  std::vector<uint32_t> indices(num_indices);
  std::vector<Vector3<float>> positions(num_vertices), normals;
  std::vector<Vector2<float>> tex_coords;
  std::vector<Vector3<float>> colors;
  stream.read(reinterpret_cast<char *>(indices.data()), indices.size() * sizeof(uint32_t));
  stream.read(reinterpret_cast<char *>(positions.data()), positions.size() * sizeof(Vector3<float>));
  if (flags & 1u) {
    normals.resize(num_vertices);
    stream.read(reinterpret_cast<char *>(normals.data()), normals.size() * sizeof(Vector3<float>));
  }
  if (flags & 2u) {
    tex_coords.resize(num_vertices);
    stream.read(reinterpret_cast<char *>(tex_coords.data()), tex_coords.size() * sizeof(Vector2<float>));
  }
  if (flags & 4u) {
    colors.resize(num_vertices);
    stream.read(reinterpret_cast<char *>(colors.data()), colors.size() * sizeof(Vector3<float>));
  }
  if (!stream)
    throw std::runtime_error("cannot read binary mesh: " + path.string());
  return Mesh<>(positions.size(), indices.size(), indices.data(), positions.data(),
                normals.empty() ? nullptr : normals.data(), tex_coords.empty() ? nullptr : tex_coords.data(), nullptr,
                colors.empty() ? nullptr : colors.data());
}

struct BinaryHairData {
  std::vector<Vector3<float>> points;
  std::vector<float> radii;
  std::vector<uint32_t> offsets;
};

BinaryHairData BinaryHair(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream)
    throw std::runtime_error("cannot open binary hair: " + path.string());
  const auto size = stream.tellg();
  stream.seekg(0);
  std::array<char, 8> magic{};
  uint32_t strand_count = 0, point_count = 0;
  stream.read(magic.data(), magic.size());
  stream.read(reinterpret_cast<char *>(&strand_count), sizeof(strand_count));
  stream.read(reinterpret_cast<char *>(&point_count), sizeof(point_count));
  const uint64_t expected = 16ull + uint64_t(strand_count + 1) * sizeof(uint32_t) +
                            uint64_t(point_count) * sizeof(Vector3<float>) + uint64_t(point_count) * sizeof(float);
  if (!stream || std::memcmp(magic.data(), "SPKHAIR1", 8) != 0 || uint64_t(size) != expected)
    throw std::runtime_error("invalid Sparkium binary hair: " + path.string());
  BinaryHairData result;
  result.offsets.resize(strand_count + 1);
  result.points.resize(point_count);
  result.radii.resize(point_count);
  stream.read(reinterpret_cast<char *>(result.offsets.data()), result.offsets.size() * sizeof(uint32_t));
  stream.read(reinterpret_cast<char *>(result.points.data()), result.points.size() * sizeof(Vector3<float>));
  stream.read(reinterpret_cast<char *>(result.radii.data()), result.radii.size() * sizeof(float));
  if (!stream || result.offsets.empty() || result.offsets.front() != 0 || result.offsets.back() != point_count)
    throw std::runtime_error("invalid Sparkium hair offsets: " + path.string());
  return result;
}

NodeValue ReadNodeValue(const Value &value) {
  if (value.IsNull())
    return {};
  if (value.IsBool())
    return {value.GetBool()};
  if (value.IsNumber())
    return {double(ReadNumber(value))};
  if (value.IsString())
    return {ReadString(value)};
  if (value.IsArray()) {
    NodeValue::Array array;
    for (const auto &item : value.GetArray())
      array.push_back(ReadNodeValue(item));
    return {std::move(array)};
  }
  NodeValue::Object object;
  for (auto it = value.MemberBegin(); it != value.MemberEnd(); ++it)
    object.emplace(ReadString(it->name), ReadNodeValue(it->value));
  return {std::move(object)};
}
}  // namespace

SceneDocument LoadSceneDocument(const std::filesystem::path &input_path) {
  RenderPipeline preferred_pipeline = RENDER_PIPELINE_AUTO;
  using namespace grassland;
  auto path = std::filesystem::absolute(input_path).lexically_normal();
  std::ifstream stream(path);
  if (!stream)
    throw std::runtime_error("cannot open scene file: " + path.string());
  rapidjson::IStreamWrapper wrapper(stream);
  rapidjson::Document document;
  document.ParseStream<rapidjson::kParseCommentsFlag | rapidjson::kParseTrailingCommasFlag>(wrapper);
  if (document.HasParseError()) {
    std::ostringstream message;
    message << "JSON parse error at byte " << document.GetErrorOffset() << ": "
            << rapidjson::GetParseError_En(document.GetParseError());
    throw std::runtime_error(message.str());
  }
  if (!document.IsObject())
    throw std::runtime_error("scene root must be an object");
  if (!document.HasMember("format") || ReadString(document["format"]) != "sparkium-scene")
    throw std::runtime_error("unsupported or missing format (expected 'sparkium-scene')");
  if (!document.HasMember("version") || ReadInt(document["version"]) != 1)
    throw std::runtime_error("unsupported scene version (expected 1)");

  auto result = std::make_shared<SceneDefinition>();
  result->name = document.HasMember("name") ? ReadString(document["name"]) : path.stem().string();
  std::map<std::filesystem::path, std::shared_ptr<const TextureData>> texture_cache;
  auto load_texture = [&](const std::string &name) -> std::shared_ptr<const TextureData> {
    auto asset = Resolve(path, name);
    if (auto it = texture_cache.find(asset); it != texture_cache.end())
      return it->second;
    auto texture = std::make_shared<TextureData>();
    int channels{};
    std::unique_ptr<unsigned char, decltype(&stbi_image_free)> bytes(
        stbi_load(asset.string().c_str(), &texture->width, &texture->height, &channels, 4), stbi_image_free);
    if (!bytes)
      throw std::runtime_error("cannot load texture: " + asset.string());
    texture->rgba.assign(bytes.get(), bytes.get() + size_t(texture->width) * texture->height * 4);
    texture_cache.emplace(asset, texture);
    return texture;
  };
  const auto &renderer = RequireObject(Member(document, "renderer"));
  result->integrator.samples_per_dispatch =
      renderer.HasMember("samples_per_dispatch") ? ReadInt(renderer["samples_per_dispatch"]) : 32;
  result->integrator.max_bounces = renderer.HasMember("max_bounces") ? ReadInt(renderer["max_bounces"]) : 32;
  if (result->integrator.samples_per_dispatch <= 0 || result->integrator.max_bounces <= 0)
    throw std::runtime_error("samples_per_dispatch and max_bounces must be positive");
  result->integrator.alpha_shadow = BoolMember(renderer, "alpha_shadow", false);
  result->integrator.ambient_light = Vec3Member(renderer, "ambient_light", {0.1f, 0.1f, 0.1f});
  result->integrator.background_color = Vec3Member(renderer, "background_color", result->integrator.ambient_light);
  std::string pipeline = renderer.HasMember("pipeline") ? ReadString(renderer["pipeline"]) : "auto";
  if (pipeline == "rasterization")
    preferred_pipeline = RENDER_PIPELINE_RASTERIZATION;
  else if (pipeline == "ray_tracing")
    preferred_pipeline = RENDER_PIPELINE_RAY_TRACING;
  else if (pipeline == "ray_query")
    preferred_pipeline = RENDER_PIPELINE_RAY_QUERY;
  else if (pipeline == "rt_fallback")
    preferred_pipeline = RENDER_PIPELINE_RT_FALLBACK;
  else if (pipeline == "auto")
    preferred_pipeline = RENDER_PIPELINE_AUTO;
  else
    throw std::runtime_error("renderer.pipeline must be auto, rasterization, ray_tracing, rt_fallback, or ray_query");

  const auto &film = RequireObject(Member(document, "film"));
  int width = ReadInt(Member(film, "width"));
  int height = ReadInt(Member(film, "height"));
  if (width <= 0 || height <= 0)
    throw std::runtime_error("film dimensions must be positive");
  result->film.width = width;
  result->film.height = height;
  result->film.persistence = FloatMember(film, "persistence", 1.0f);
  result->film.clamping = FloatMember(film, "clamping", 100.0f);
  result->film.max_exposure = FloatMember(film, "max_exposure", 1.0f);
  std::string view_transform = film.HasMember("view_transform") ? ReadString(film["view_transform"]) : "normalized";
  if (view_transform == "normalized")
    result->film.view_transform = 0;
  else if (view_transform == "standard")
    result->film.view_transform = 1;
  else if (view_transform == "filmic")
    result->film.view_transform = 2;
  else
    throw std::runtime_error("film.view_transform must be normalized, standard, or filmic");
  result->film.exposure = FloatMember(film, "exposure", 0.0f);
  result->film.gamma = FloatMember(film, "gamma", 1.0f);
  result->film.contrast = FloatMember(film, "contrast", 1.0f);

  const auto &camera = RequireObject(Member(document, "camera"));
  auto eye = Vec3Member(camera, "eye");
  auto target = Vec3Member(camera, "target");
  auto up = Vec3Member(camera, "up", {0.0f, 1.0f, 0.0f});
  float fov = FloatMember(camera, "fov_degrees", 60.0f);
  if (!(fov > 0.0f && fov < 180.0f) || glm::length(target - eye) < 1e-6f ||
      glm::length(glm::cross(target - eye, up)) < 1e-6f)
    throw std::runtime_error("camera requires a valid field of view and nondegenerate look-at vectors");
  result->camera.view = glm::lookAt(eye, target, up);
  result->camera.fovy = glm::radians(fov);
  result->camera.aspect = static_cast<float>(width) / height;
  result->camera.aperture_radius = FloatMember(camera, "aperture_radius", 0.0f);
  result->camera.focus_distance = FloatMember(camera, "focus_distance", glm::length(target - eye));
  result->camera.aperture_blades = camera.HasMember("aperture_blades") ? ReadInt(camera["aperture_blades"]) : 0;
  result->camera.aperture_rotation = FloatMember(camera, "aperture_rotation", 0.0f);
  result->camera.aperture_ratio = FloatMember(camera, "aperture_ratio", 1.0f);

  const auto &materials = RequireObject(Member(document, "materials"));
  for (auto it = materials.MemberBegin(); it != materials.MemberEnd(); ++it) {
    std::string id = ReadString(it->name);
    const auto &spec = RequireObject(it->value);
    std::string type = ReadString(Member(spec, "type"));
    MaterialDefinition material;
    material.base_color = Vec3Member(spec, "base_color", glm::vec3{0.8f});
    material.emission = Vec3Member(spec, "emission");
    if (type == "lambertian")
      material.type = MaterialDefinition::Type::Lambertian;
    else if (type == "specular")
      material.type = MaterialDefinition::Type::Specular;
    else if (type == "light") {
      material.type = MaterialDefinition::Type::Light;
      material.two_sided = BoolMember(spec, "two_sided", false);
      material.block_ray = BoolMember(spec, "block_ray", false);
      material.camera_visible = BoolMember(spec, "camera_visible", true);
      material.falloff_distance = FloatMember(spec, "falloff_distance", 0.0f);
    } else if (type == "principled") {
      material.type = MaterialDefinition::Type::Principled;
      auto *principled = &material.principled;
      principled->base_color = material.base_color;
      principled->subsurface_color = Vec3Member(spec, "subsurface_color", principled->subsurface_color);
      principled->subsurface_radius = Vec3Member(spec, "subsurface_radius", principled->subsurface_radius);
#define READ_FLOAT(field) principled->field = FloatMember(spec, #field, principled->field)
      READ_FLOAT(subsurface);
      READ_FLOAT(metallic);
      READ_FLOAT(specular);
      READ_FLOAT(specular_tint);
      READ_FLOAT(roughness);
      READ_FLOAT(anisotropic);
      READ_FLOAT(anisotropic_rotation);
      READ_FLOAT(sheen);
      READ_FLOAT(sheen_tint);
      READ_FLOAT(clearcoat);
      READ_FLOAT(clearcoat_roughness);
      READ_FLOAT(ior);
      READ_FLOAT(transmission);
      READ_FLOAT(transmission_roughness);
      READ_FLOAT(emission_strength);
#undef READ_FLOAT
      principled->emission_color = Vec3Member(spec, "emission_color", principled->emission_color);
      if (spec.HasMember("textures")) {
        const auto &textures = RequireObject(spec["textures"]);
        for (const char *slot : {"normal", "base_color", "metallic", "specular", "roughness", "anisotropic",
                                 "anisotropic_rotation", "emission"})
          if (textures.HasMember(slot))
            material.textures[slot] = load_texture(ReadString(textures[slot]));
        material.normal_reverse_y = BoolMember(textures, "normal_reverse_y", false);
      }
    } else if (type == "shader_graph") {
      material.type = MaterialDefinition::Type::ShaderGraph;
      const auto &graph = RequireObject(Member(spec, "graph"));
      const auto &nodes = RequireObject(Member(graph, "nodes"));
      material.graph = ReadNodeValue(graph);
      for (auto node = nodes.MemberBegin(); node != nodes.MemberEnd(); ++node) {
        if (!RequireObject(node->value).HasMember("type") || ReadString(node->value["type"]) != "image_texture")
          continue;
        auto id = ReadString(node->name);
        material.graph_textures[id] = load_texture(ReadString(Member(node->value, "path")));
        // The path belongs to the loader, never to the in-memory node graph.
        auto &graph_object = std::get<NodeValue::Object>(material.graph.value);
        auto &memory_nodes = std::get<NodeValue::Object>(graph_object.at("nodes").value);
        std::get<NodeValue::Object>(memory_nodes.at(id).value).erase("path");
      }
      material.emission_hint = Vec3Member(spec, "emission_hint");
    } else
      throw std::runtime_error("unknown material type: " + type);
    result->materials.emplace(id, std::move(material));
  }
  const auto &geometries = RequireObject(Member(document, "geometries"));
  for (auto it = geometries.MemberBegin(); it != geometries.MemberEnd(); ++it) {
    std::string id = ReadString(it->name);
    const auto &spec = RequireObject(it->value);
    std::string type = ReadString(Member(spec, "type"));
    if (type == "hair") {
      auto hair = BinaryHair(Resolve(path, ReadString(Member(spec, "path"))));
      int radial_segments = spec.HasMember("radial_segments") ? ReadInt(spec["radial_segments"]) : 3;
      auto geometry = std::make_shared<GeometryDefinition>();
      geometry->shape =
          HairData{std::move(hair.points), std::move(hair.radii), std::move(hair.offsets), radial_segments};
      result->geometries.emplace(id, std::move(geometry));
      continue;
    }
    Mesh<> mesh;
    if (type == "sphere") {
      int longitude = spec.HasMember("longitude_segments") ? ReadInt(spec["longitude_segments"]) : 30;
      int latitude = spec.HasMember("latitude_segments") ? ReadInt(spec["latitude_segments"]) : -1;
      if (longitude < 3 || (latitude != -1 && latitude < 2))
        throw std::runtime_error("sphere requires at least 3 longitude and 2 latitude segments");
      mesh = Mesh<>::Sphere(longitude, latitude);
    } else if (type == "mesh") {
      auto asset = Resolve(path, ReadString(Member(spec, "path")));
      if (mesh.LoadObjFile(asset.string()) != 0)
        throw std::runtime_error("cannot load mesh: " + asset.string());
    } else if (type == "binary_mesh") {
      mesh = BinaryMesh(Resolve(path, ReadString(Member(spec, "path"))));
    } else if (type == "inline_mesh") {
      mesh = InlineMesh(spec);
    } else {
      throw std::runtime_error("unknown geometry type: " + type);
    }
    if (BoolMember(spec, "generate_normals", false))
      mesh.GenerateNormals();
    if (BoolMember(spec, "generate_tangents", false))
      mesh.GenerateTangents();
    auto geometry = std::make_shared<GeometryDefinition>();
    geometry->shape = std::move(mesh);
    result->geometries.emplace(id, std::move(geometry));
  }

  const auto &entities = RequireArray(Member(document, "entities"));
  for (const auto &spec : entities.GetArray()) {
    std::string type = ReadString(Member(spec, "type"));
    if (type == "mesh") {
      std::string geometry_id = ReadString(Member(spec, "geometry"));
      std::string material_id = ReadString(Member(spec, "material"));
      auto geometry = result->geometries.find(geometry_id);
      auto material = result->materials.find(material_id);
      if (geometry == result->geometries.end())
        throw std::runtime_error("unknown geometry: " + geometry_id);
      if (material == result->materials.end())
        throw std::runtime_error("unknown material: " + material_id);
      glm::mat4 transform{1.0f};
      if (spec.HasMember("transform"))
        transform = Transform(spec["transform"]);
      if (spec.HasMember("look_at")) {
        const auto &look = RequireObject(spec["look_at"]);
        auto position = Vec3Member(look, "position");
        auto target = Vec3Member(look, "target");
        auto up = Vec3Member(look, "up", {0.0f, 1.0f, 0.0f});
        auto scale = Vec3Member(look, "scale", glm::vec3{1.0f});
        transform = glm::inverse(glm::lookAt(position, target, up)) * glm::scale(glm::mat4{1.0f}, scale);
      }
      result->entities.emplace_back(
          InstanceDefinition{geometry_id, material_id, transform, BoolMember(spec, "raster_light", true)});
    } else if (type == "point_light") {
      PointLightDefinition light;
      light.position = Vec3Member(spec, "position");
      light.color = Vec3Member(spec, "color", glm::vec3{1.0f});
      light.strength = FloatMember(spec, "strength", 0.0f);
      light.radius = FloatMember(spec, "radius", 0.0f);
      light.soft_falloff = BoolMember(spec, "soft_falloff", false);
      light.sampling_weight = FloatMember(spec, "sampling_weight", -1.0f);
      result->entities.emplace_back(light);
    } else {
      throw std::runtime_error("unknown entity type: " + type);
    }
  }
  result->Validate();
  return {result, preferred_pipeline};
}

std::shared_ptr<const SceneDefinition> LoadScene(const std::filesystem::path &path) {
  return LoadSceneDocument(path).scene;
}

std::vector<std::filesystem::path> FindJsonScenes(const std::filesystem::path &directory) {
  std::vector<std::filesystem::path> result;
  if (!std::filesystem::exists(directory))
    return result;
  for (const auto &entry : std::filesystem::recursive_directory_iterator(directory)) {
    if (entry.is_regular_file() && entry.path().filename() == "scene.json")
      result.push_back(entry.path());
  }

  std::sort(result.begin(), result.end());
  return result;
}
}  // namespace sparkium
