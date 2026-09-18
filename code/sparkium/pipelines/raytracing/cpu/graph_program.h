// Runtime evaluation of shader-graph materials on the CPU.
//
// A shader graph's body is generated per scene at load time
// (ShaderGraphCompiler in scene_io/json_scene.cpp), so unlike every other
// material it cannot be compiled into the backend ahead of time. Everything
// around it can: the sampler, the surface parameters and the BSDFs are all
// statically compiled, and the single generated function EvaluateShaderGraph
// is reached through this interface instead.
//
// The GPU backends do the same thing with DXC at runtime. Here the generated
// source is evaluated by one of two engines -- a tree-walking interpreter and
// an in-process Clang JIT -- which share this interface and are expected to
// agree exactly. See graph_interpreter.h and graph_jit.h.
//
// The interface is deliberately free of the HLSL compatibility layer's types:
// the CPU backend's shader translation unit owns those, and a graph engine is
// an ordinary translation unit that only sees plain arrays and bytes.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sparkium::raytracing::cpu {

// Everything the generated function reads, in plain arrays so an engine never
// has to know the shader's struct layouts.
struct GraphEvalInput {
  float position[3]{};
  float object_position[3]{};
  // The object's world space origin, which is not the object space hit
  // position; HitRecord keeps both and the object_info node reads this one.
  float object_origin[3]{};
  float normal[3]{};
  float geom_normal[3]{};
  float tangent[3]{};
  float tex_coord[2]{};
  float color[3]{};
  float t{0.0f};
  float signal{0.0f};
  float pdf{0.0f};
  int32_t primitive_index{0};
  int32_t object_index{0};
  bool front_facing{true};

  float view_direction[3]{};
  int32_t bounce{0};
  int32_t ray_type{0};
  bool is_shadow_ray{false};

  const uint8_t *material_data{nullptr};
  size_t material_data_size{0};

  // SampleTexture from bindings.hlsli, which reads the same texture tables the
  // path tracer binds.
  void (*sample_texture)(void *user, int32_t texture_index, float u, float v, float out[4]){nullptr};
  void *sample_texture_user{nullptr};
};

// The result of EvaluateShaderGraph. The fields are named rather than a flat
// array so neither engine has to agree with the shader's struct layout; the
// glue in cpu_shaders.cpp assigns them one by one.
struct GraphEvalOutput {
  float base_color[3]{};
  float metallic{0.0f};
  float specular{0.0f};
  float roughness{0.0f};
  float anisotropic{0.0f};
  float anisotropic_rotation{0.0f};
  float sheen{0.0f};
  float clearcoat{0.0f};
  float clearcoat_roughness{0.0f};
  float ior{0.0f};
  float transmission{0.0f};
  float transmission_roughness{0.0f};
  float emission[3]{};
  float normal[3]{};
  float opacity{0.0f};
  float shadow_opacity{0.0f};
  float thin_walled{0.0f};
  float subsurface{0.0f};
  float subsurface_scale{0.0f};
  float subsurface_radius[3]{};
  float subsurface_method{0.0f};
};

// Evaluates one material's generated graph code.
class GraphProgram {
 public:
  virtual ~GraphProgram() = default;

  virtual void Evaluate(const GraphEvalInput &input, GraphEvalOutput &output) const = 0;

  // Human readable engine name, for diagnostics and the CLI.
  virtual const char *Engine() const = 0;
};

// Compiles `source` (the text ShaderGraphCompiler produced, exactly as the GPU
// backend writes it into software_materials.hlsli). Returns nullptr and logs
// when the source cannot be handled by the requested engine.
std::unique_ptr<GraphProgram> MakeInterpreterGraphProgram(const std::string &source);
std::unique_ptr<GraphProgram> MakeJitGraphProgram(const std::string &source);

// The JIT compiles every graph of a scene in one module, because they share a
// large preamble. `sdr_views`/`hdr_views` are arrays of TextureView-shaped
// records, passed as void so this header stays independent of cpu_shaders.h.
std::vector<std::unique_ptr<GraphProgram>> MakeJitGraphPrograms(const std::vector<std::string> &sources,
                                                                const void *sdr_views,
                                                                int sdr_count,
                                                                const void *hdr_views,
                                                                int hdr_count);

// True when this build has the JIT; without it only the interpreter is offered.
bool JitGraphProgramsAvailable();

// Holds one program per material data buffer slot. Materials are keyed by the
// buffer index the shader reads out of InstanceMetadata, which is unique per
// material.
class GraphProgramRegistry {
 public:
  void Register(uint32_t material_data_index, std::unique_ptr<GraphProgram> program);
  void Clear();
  bool HasProgram(uint32_t material_data_index) const;

  // Leaves `output` at its defaults when no program is registered, and reports
  // once per material, so a missing registration is visible rather than
  // silently producing a black surface.
  void Evaluate(uint32_t material_data_index, const GraphEvalInput &input, GraphEvalOutput &output) const;

 private:
  std::vector<std::unique_ptr<GraphProgram>> programs_;
  mutable std::vector<bool> reported_;
};

}  // namespace sparkium::raytracing::cpu
