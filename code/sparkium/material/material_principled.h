#pragma once
#include "sparkium/core/material.h"

namespace sparkium {

class MaterialPrincipled : public Material {
 public:
  MaterialPrincipled(Core *core, const glm::vec3 &base_color = glm::vec3{0.8f});

  struct Info {
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
  } info;

  struct TextureInfo {
    graphics::Image *normal{nullptr};
    bool normal_reverse_y{false};
    graphics::Image *base_color{nullptr};
    graphics::Image *metallic{nullptr};
    graphics::Image *specular{nullptr};
    graphics::Image *roughness{nullptr};
    graphics::Image *anisotropic{nullptr};
    graphics::Image *anisotropic_rotation{nullptr};
  } textures{};

  glm::vec3 &base_color{info.base_color};
  glm::vec3 &subsurface_color{info.subsurface_color};
  float &subsurface{info.subsurface};
  glm::vec3 &subsurface_radius{info.subsurface_radius};
  float &metallic{info.metallic};
  float &specular{info.specular};
  float &specular_tint{info.specular_tint};
  float &roughness{info.roughness};
  float &anisotropic{info.anisotropic};

  float &anisotropic_rotation{info.anisotropic_rotation};
  float &sheen{info.sheen};
  float &sheen_tint{info.sheen_tint};
  float &clearcoat{info.clearcoat};

  float &clearcoat_roughness{info.clearcoat_roughness};
  float &ior{info.ior};
  float &transmission{info.transmission};
  float &transmission_roughness{info.transmission_roughness};

  glm::vec3 &emission_color{info.emission_color};
  float &emission_strength{info.emission_strength};
};

}  // namespace sparkium
