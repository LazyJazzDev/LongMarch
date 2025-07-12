#pragma once
#include "sparks/core/core_util.h"

namespace sparks {

class Scene {
 public:
  Scene(Core *core);

  void Render(Camera *camera, Film *film);

 private:
  void UpdatePipeline();
  Core *core_;
  std::unique_ptr<graphics::Shader> raygen_shader_;
  std::unique_ptr<graphics::RayTracingProgram> rt_program_;
  std::unique_ptr<graphics::AccelerationStructure> tlas_;

  std::vector<int32_t> miss_shader_indices_;
  std::vector<int32_t> hit_group_indices_;
  std::vector<int32_t> callable_shader_indices_;
};

}  // namespace sparks
