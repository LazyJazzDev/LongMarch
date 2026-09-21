#pragma once
#include "sparkium/core/material.h"
#include "sparkium/scene/scene_definition.h"

namespace sparkium {

class MaterialPrincipled : public Material {
 public:
  MaterialPrincipled(Core *core, const glm::vec3 &base_color = glm::vec3{0.8f});

  using Info = PrincipledParameters;
  Info info;

  struct TextureInfo {
    graphics::Image *normal{nullptr};
    bool normal_reverse_y{false};
    graphics::Image *base_color{nullptr};
    graphics::Image *metallic{nullptr};
    graphics::Image *specular{nullptr};
    graphics::Image *roughness{nullptr};
    graphics::Image *anisotropic{nullptr};
    graphics::Image *anisotropic_rotation{nullptr};
    graphics::Image *emission{nullptr};
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
