#include "sparkium/scene_io/json_scene.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "glm/gtc/matrix_transform.hpp"
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "rapidjson/istreamwrapper.h"
#include "sparkium/core/core.h"

namespace sparkium {
namespace {
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
  return glm::scale(result, Vec3Member(value, "scale", {1.0f, 1.0f, 1.0f}));
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
}  // namespace

std::unique_ptr<JsonScene> JsonScene::Load(Core *core, const std::filesystem::path &input_path, std::string *error) {
  try {
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

    auto result = std::unique_ptr<JsonScene>(new JsonScene);
    result->core_ = core;
    result->path_ = path;
    result->name_ = document.HasMember("name") ? ReadString(document["name"]) : path.stem().string();
    result->scene_ = std::make_unique<Scene>(core);

    const auto &renderer = RequireObject(Member(document, "renderer"));
    result->scene_->settings.raytracing.samples_per_dispatch =
        renderer.HasMember("samples_per_dispatch") ? ReadInt(renderer["samples_per_dispatch"]) : 32;
    result->scene_->settings.raytracing.max_bounces =
        renderer.HasMember("max_bounces") ? ReadInt(renderer["max_bounces"]) : 32;
    if (result->scene_->settings.raytracing.samples_per_dispatch <= 0 ||
        result->scene_->settings.raytracing.max_bounces <= 0)
      throw std::runtime_error("samples_per_dispatch and max_bounces must be positive");
    result->scene_->settings.raytracing.alpha_shadow = BoolMember(renderer, "alpha_shadow", false);
    result->scene_->settings.raster.ambient_light = Vec3Member(renderer, "ambient_light", {0.1f, 0.1f, 0.1f});
    std::string pipeline = renderer.HasMember("pipeline") ? ReadString(renderer["pipeline"]) : "auto";
    if (pipeline == "rasterization")
      result->render_pipeline_ = RENDER_PIPELINE_RASTERIZATION;
    else if (pipeline == "ray_tracing")
      result->render_pipeline_ = RENDER_PIPELINE_RAY_TRACING;
    else if (pipeline == "ray_query")
      result->render_pipeline_ = RENDER_PIPELINE_RAY_QUERY;
    else if (pipeline == "rt_fallback")
      result->render_pipeline_ = RENDER_PIPELINE_RT_FALLBACK;
    else if (pipeline == "auto")
      result->render_pipeline_ = RENDER_PIPELINE_AUTO;
    else
      throw std::runtime_error("renderer.pipeline must be auto, rasterization, ray_tracing, rt_fallback, or ray_query");

    const auto &film = RequireObject(Member(document, "film"));
    int width = ReadInt(Member(film, "width"));
    int height = ReadInt(Member(film, "height"));
    if (width <= 0 || height <= 0)
      throw std::runtime_error("film dimensions must be positive");
    result->film_ = std::make_unique<Film>(core, width, height);
    result->film_->info.persistence = FloatMember(film, "persistence", 1.0f);
    result->film_->info.clamping = FloatMember(film, "clamping", 100.0f);
    result->film_->info.max_exposure = FloatMember(film, "max_exposure", 1.0f);

    const auto &camera = RequireObject(Member(document, "camera"));
    auto eye = Vec3Member(camera, "eye");
    auto target = Vec3Member(camera, "target");
    auto up = Vec3Member(camera, "up", {0.0f, 1.0f, 0.0f});
    float fov = FloatMember(camera, "fov_degrees", 60.0f);
    if (!(fov > 0.0f && fov < 180.0f) || glm::length(target - eye) < 1e-6f ||
        glm::length(glm::cross(target - eye, up)) < 1e-6f)
      throw std::runtime_error("camera requires a valid field of view and nondegenerate look-at vectors");
    result->camera_ = std::make_unique<Camera>(core, glm::lookAt(eye, target, up), glm::radians(fov),
                                               static_cast<float>(width) / static_cast<float>(height));

    const auto &materials = RequireObject(Member(document, "materials"));
    for (auto it = materials.MemberBegin(); it != materials.MemberEnd(); ++it) {
      std::string id = ReadString(it->name);
      const auto &spec = RequireObject(it->value);
      std::string type = ReadString(Member(spec, "type"));
      std::unique_ptr<Material> material;
      if (type == "lambertian") {
        material = std::make_unique<MaterialLambertian>(core, Vec3Member(spec, "base_color", {0.8f, 0.8f, 0.8f}),
                                                        Vec3Member(spec, "emission"));
      } else if (type == "specular") {
        material = std::make_unique<MaterialSpecular>(core, Vec3Member(spec, "base_color", {0.8f, 0.8f, 0.8f}));
      } else if (type == "light") {
        material =
            std::make_unique<MaterialLight>(core, Vec3Member(spec, "emission"), BoolMember(spec, "two_sided", false),
                                            BoolMember(spec, "block_ray", false));
      } else if (type == "principled") {
        auto principled = std::make_unique<MaterialPrincipled>(core, Vec3Member(spec, "base_color", glm::vec3{0.8f}));
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
          const std::pair<const char *, graphics::Image **> slots[] = {
              {"normal", &principled->textures.normal},
              {"base_color", &principled->textures.base_color},
              {"metallic", &principled->textures.metallic},
              {"specular", &principled->textures.specular},
              {"roughness", &principled->textures.roughness},
              {"anisotropic", &principled->textures.anisotropic},
              {"anisotropic_rotation", &principled->textures.anisotropic_rotation}};
          for (auto [slot, destination] : slots) {
            if (!textures.HasMember(slot))
              continue;
            auto image = std::unique_ptr<graphics::Image>{};
            auto asset = Resolve(path, ReadString(textures[slot]));
            if (graphics::LoadImageFromFile(core->GraphicsCore(), asset.string(), &image) != 0)
              throw std::runtime_error("cannot load texture: " + asset.string());
            *destination = image.get();
            result->images_.push_back(std::move(image));
          }
          principled->textures.normal_reverse_y = BoolMember(textures, "normal_reverse_y", false);
        }
        material = std::move(principled);
      } else {
        throw std::runtime_error("unknown material type: " + type);
      }
      result->materials_.emplace(id, std::move(material));
    }

    const auto &geometries = RequireObject(Member(document, "geometries"));
    for (auto it = geometries.MemberBegin(); it != geometries.MemberEnd(); ++it) {
      std::string id = ReadString(it->name);
      const auto &spec = RequireObject(it->value);
      std::string type = ReadString(Member(spec, "type"));
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
      } else if (type == "inline_mesh") {
        mesh = InlineMesh(spec);
      } else {
        throw std::runtime_error("unknown geometry type: " + type);
      }
      if (BoolMember(spec, "generate_normals", false))
        mesh.GenerateNormals();
      if (BoolMember(spec, "generate_tangents", false))
        mesh.GenerateTangents();
      result->geometries_.emplace(id, std::make_unique<GeometryMesh>(core, mesh));
    }

    const auto &entities = RequireArray(Member(document, "entities"));
    for (const auto &spec : entities.GetArray()) {
      std::string type = ReadString(Member(spec, "type"));
      Entity *entity_ptr = nullptr;
      if (type == "mesh") {
        auto geometry = result->geometries_.find(ReadString(Member(spec, "geometry")));
        auto material = result->materials_.find(ReadString(Member(spec, "material")));
        if (geometry == result->geometries_.end())
          throw std::runtime_error("unknown geometry");
        if (material == result->materials_.end())
          throw std::runtime_error("unknown material");
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
        auto entity =
            std::make_unique<EntityGeometryMaterial>(core, geometry->second.get(), material->second.get(), transform);
        entity->raster_light = BoolMember(spec, "raster_light", true);
        entity_ptr = entity.get();
        result->entities_.push_back(std::move(entity));
      } else if (type == "point_light") {
        auto entity = std::make_unique<EntityPointLight>(core, Vec3Member(spec, "position"),
                                                         Vec3Member(spec, "color", glm::vec3{1.0f}),
                                                         FloatMember(spec, "strength", 0.0f));
        entity_ptr = entity.get();
        result->entities_.push_back(std::move(entity));
      } else {
        throw std::runtime_error("unknown entity type: " + type);
      }
      result->scene_->AddEntity(entity_ptr);
    }
    return result;
  } catch (const std::exception &exception) {
    if (error)
      *error = exception.what();
    return nullptr;
  }
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
