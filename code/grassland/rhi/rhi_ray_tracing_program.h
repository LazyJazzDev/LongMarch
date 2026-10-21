#pragma once
#include "grassland/rhi/rhi_program.h"

namespace grassland::rhi {
struct RayTracingHitGroup {
  IShader *closest_hit{};
  IShader *any_hit{};
  IShader *intersection{};
};

// The default shader table order follows the vectors. Hit groups without an intersection
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

// Indices address the corresponding arrays in RayTracingProgramDesc.
// Repetition and omission are allowed; empty arrays produce empty table regions.
// Records contain shader identifiers only, without local record data.
// The single ray-generation record always uses the program's ray_generation.
struct ShaderBindingTableDesc {
  std::vector<uint32_t> miss_shader_indices;
  std::vector<uint32_t> hit_group_indices;
  std::vector<uint32_t> callable_shader_indices;
};

class IRayTracingProgram;

class IShaderBindingTable : public IObject {
 public:
  virtual const ShaderBindingTableDesc &GetDesc() const = 0;
  // Borrowed parent; release caller-owned tables before their program.
  virtual IRayTracingProgram *GetProgram() const = 0;

 protected:
  ~IShaderBindingTable() override = default;
};

class IRayTracingProgram : public IObject {
 public:
  // Requires a finalized program. Returns one owned, immutable table.
  virtual IShaderBindingTable *CreateShaderBindingTable(const ShaderBindingTableDesc &desc) = 0;
  virtual bool Finalize() = 0;
  virtual bool IsFinalized() const = 0;
  virtual const RayTracingProgramDesc &GetDesc() const = 0;

 protected:
  ~IRayTracingProgram() override;
};
}  // namespace grassland::rhi
