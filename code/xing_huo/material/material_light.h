#pragma once
#include "xing_huo/core/material.h"

namespace XH {

class MaterialLight : public Material {
 public:
  MaterialLight(Core *core,
                const glm::vec3 &emission = glm::vec3{0.0f},
                bool two_sided = false,
                bool block_ray = false);

  graphics::Buffer *Buffer() override;
  const CodeLines &SamplerImpl() const override;
  const CodeLines &EvaluatorImpl() const override;

  glm::vec3 emission{0.0f};
  int two_sided = 0;
  int block_ray = 0;

  void SyncMaterialData();

 private:
  std::unique_ptr<graphics::Buffer> material_buffer_;
  CodeLines sampler_implementation_;
  CodeLines evaluator_implementation_;
};

}  // namespace XH
