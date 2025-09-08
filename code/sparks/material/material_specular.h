#pragma once
#include "sparks/core/material.h"

namespace XH {

class MaterialSpecular : public Material {
 public:
  MaterialSpecular(Core *core, const glm::vec3 &base_color = glm::vec3{0.8f});

  graphics::Buffer *Buffer() override;
  const CodeLines &SamplerImpl() const override;
  const CodeLines &EvaluatorImpl() const override;

  glm::vec3 base_color{0.8f};

  void SyncMaterialData();

 private:
  std::unique_ptr<graphics::Buffer> material_buffer_;
  CodeLines sampler_implementation_;
  CodeLines evaluator_implementation_;
};

}  // namespace XH
