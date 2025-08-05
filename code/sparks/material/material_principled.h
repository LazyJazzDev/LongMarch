#pragma once
#include "sparks/core/material.h"

namespace sparks {

class MaterialPrincipled : public Material {
 public:
  MaterialPrincipled(Core *core, const glm::vec3 &base_color = glm::vec3{0.8f});

  graphics::Buffer *Buffer() override;
  const CodeLines &SamplerImpl() const override;
  const CodeLines &EvaluatorImpl() const override;

  glm::vec3 base_color{0.8f};

  glm::vec3 subsurface_color{1.0f, 1.0f, 1.0f};
  float subsurface{0.0f};

  glm::vec3 subsurface_radius{1.0f, 0.2f, 0.1f};
  float metallic{0.05f};

  float specular{0.0f};
  float specular_tint{0.0f};
  float roughness{0.0f};
  float anisotropic{0.0f};

  float anisotropic_rotation{0.0f};
  float sheen{0.0f};
  float sheen_tint{0.0f};
  float clearcoat{0.0f};

  float clearcoat_roughness{0.0f};
  float ior{1.2f};
  float transmission{0.0f};
  float transmission_roughness{0.0f};

  void SyncMaterialData();

 private:
  std::unique_ptr<graphics::Buffer> material_buffer_;
  CodeLines sampler_implementation_;
  CodeLines evaluator_implementation_;
};

}  // namespace sparks
