#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "glm/glm.hpp"
#include "grassland/math/math_mesh.h"

namespace sparkium {

// A format-independent value tree for material node properties and connections.
// Image nodes refer to decoded textures through MaterialDefinition::graph_textures.
struct NodeValue {
  using Array = std::vector<NodeValue>;
  using Object = std::map<std::string, NodeValue>;
  std::variant<std::monostate, bool, double, std::string, Array, Object> value;
};

struct TextureData {
  int width{}, height{};
  std::vector<uint8_t> rgba;
};

struct HairData {
  std::vector<grassland::Vector3<float>> points;
  std::vector<float> radii;
  std::vector<uint32_t> offsets;
  int radial_segments{3};
};

struct GeometryDefinition {
  std::variant<grassland::Mesh<float>, HairData> shape;
};

struct PrincipledParameters {
  glm::vec3 base_color{0.8f};

  glm::vec3 subsurface_color{1.0f, 1.0f, 1.0f};
  float subsurface{0.0f};

  glm::vec3 subsurface_radius{1.0f, 0.2f, 0.1f};
  float metallic{0.0f};

  float specular{0.0f};
  float specular_tint{0.0f};
  float roughness{0.5f};
  float anisotropic{0.0f};

  float anisotropic_rotation{0.0f};
  float sheen{0.0f};
  float sheen_tint{0.0f};
  float clearcoat{0.0f};

  float clearcoat_roughness{0.0f};
  float ior{1.45f};
  float transmission{0.0f};
  float transmission_roughness{0.0f};

  glm::vec3 emission_color{1.0f};
  float emission_strength{0.0f};
};

struct MaterialDefinition {
  enum class Type { Lambertian, Specular, Light, Principled, ShaderGraph };
  Type type{Type::Lambertian};
  glm::vec3 base_color{0.8f}, emission{0.0f};
  bool two_sided{}, block_ray{}, camera_visible{true};
  float falloff_distance{};
  PrincipledParameters principled;
  std::map<std::string, std::shared_ptr<const TextureData>> textures;
  bool normal_reverse_y{};
  NodeValue graph;
  std::map<std::string, std::shared_ptr<const TextureData>> graph_textures;
  glm::vec3 emission_hint{0.0f};
};

struct InstanceDefinition {
  std::string geometry, material;
  glm::mat4 transform{1.0f};
  bool raster_light{true}, active{true};
};

struct PointLightDefinition {
  glm::vec3 position{0.0f}, color{1.0f};
  float strength{}, radius{}, sampling_weight{-1.0f};
  bool soft_falloff{}, active{true};
};

struct CameraDefinition {
  glm::mat4 view{1.0f};
  float fovy{1.04719755f}, aspect{1.0f};
  float aperture_radius{}, focus_distance{1.0f};
  int aperture_blades{};
  float aperture_rotation{}, aperture_ratio{1.0f};
};

struct FilmDefinition {
  int width{512}, height{512};
  float persistence{1.0f}, clamping{100.0f}, max_exposure{1.0f};
  int view_transform{};
  float exposure{}, gamma{1.0f}, contrast{1.0f};
};

struct IntegratorDefinition {
  int samples_per_dispatch{32}, max_bounces{32};
  bool alpha_shadow{};
  glm::vec3 background_color{0.1f}, ambient_light{0.1f};
};

// Complete, editable host data. No source paths, loader state or device resources.
// Publish as shared_ptr<const SceneDefinition> while rendering. To edit a scene,
// create a new snapshot (immutable geometry/texture storage may remain shared).
struct SceneDefinition {
  std::string name;
  FilmDefinition film;
  CameraDefinition camera;
  IntegratorDefinition integrator;
  std::map<std::string, MaterialDefinition> materials;
  std::map<std::string, std::shared_ptr<const GeometryDefinition>> geometries;
  std::vector<std::variant<InstanceDefinition, PointLightDefinition>> entities;

  void Validate() const;
};

}  // namespace sparkium
