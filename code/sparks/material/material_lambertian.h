#pragma once
#include "sparks/core/material.h"

namespace XH {

class MaterialLambertian : public Material {
 public:
  MaterialLambertian(Core *core,
                     const glm::vec3 &base_color = glm::vec3{0.8f},
                     const glm::vec3 &emission = glm::vec3{0.0f});

  graphics::Buffer *Buffer() override;
  const CodeLines &SamplerImpl() const override;
  const CodeLines &EvaluatorImpl() const override;

  glm::vec3 base_color{0.8f};
  glm::vec3 emission{0.0f};

  void SyncMaterialData();

 private:
  std::unique_ptr<graphics::Buffer> material_buffer_;
  CodeLines sampler_implementation_;
  CodeLines evaluator_implementation_;
};

}  // namespace XH
