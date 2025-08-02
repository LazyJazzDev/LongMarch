#pragma once
#include "sparks/core/surface.h"

namespace sparks {

class SurfaceLambertian : public Surface {
 public:
  SurfaceLambertian(Core *core,
                    const glm::vec3 &base_color = glm::vec3{0.8f},
                    const glm::vec3 &emission = glm::vec3{0.0f});

  graphics::Buffer *Buffer() override;
  graphics::Shader *CallableShader() override;
  const CodeLines &SamplerImpl() const override;
  const CodeLines &EvaluatorImpl() const override;

  glm::vec3 base_color{0.8f};
  glm::vec3 emission{0.0f};

  void SyncSurfaceData();

 private:
  std::unique_ptr<graphics::Buffer> surface_buffer_;
  std::unique_ptr<graphics::Shader> callable_shader_;
  CodeLines sampler_implementation_;
  CodeLines evaluator_implementation_;
};

}  // namespace sparks
