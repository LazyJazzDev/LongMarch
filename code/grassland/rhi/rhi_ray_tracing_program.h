#pragma once
#include "grassland/rhi/rhi_program.h"

namespace grassland::rhi {
struct RayTracingHitGroup {
  IShader *closest_hit{};
  IShader *any_hit{};
  IShader *intersection{};
};

// Shader table order follows the vectors. Hit groups without an intersection
// shader describe triangles; an intersection shader selects procedural geometry.
struct RayTracingProgramDesc {
  IShader *ray_generation{};
  std::vector<IShader *> miss_shaders;
  std::vector<RayTracingHitGroup> hit_groups;
  std::vector<IShader *> callable_shaders;
  PipelineLayoutDesc pipeline_layout;
  uint32_t max_recursion_depth{1};
  uint32_t max_payload_size{32};
  uint32_t max_attribute_size{8};
};

class IRayTracingProgram : public IObject {
 public:
  virtual bool Finalize() = 0;
  virtual bool IsFinalized() const = 0;
  virtual const RayTracingProgramDesc &GetDesc() const = 0;

 protected:
  ~IRayTracingProgram() override;
};
}  // namespace grassland::rhi
