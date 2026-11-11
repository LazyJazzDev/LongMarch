#pragma once
#include "grassland/rhi/rhi_program.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::IRayTracingProgram,
                        grassland::rhi::IObject,
                        "cc23cbda-1b3c-4fb6-9bf6-c04dd82f6f6d")
LONGMARCH_RHI_INTERFACE(grassland::rhi::IShaderBindingTable,
                        grassland::rhi::IObject,
                        "50687db6-c26d-4f57-82c9-7bd6ae21d68e")

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

// Creation compiles the pipeline and default SBT; successful objects are ready to bind.
class IRayTracingProgram : public IObject {
 public:
  // Returns one owned, immutable table, or nullptr on failure.
  virtual IShaderBindingTable *CreateShaderBindingTable(const ShaderBindingTableDesc &desc) = 0;
  virtual const RayTracingProgramDesc &GetDesc() const = 0;

 protected:
  ~IRayTracingProgram() override;
};
}  // namespace grassland::rhi
