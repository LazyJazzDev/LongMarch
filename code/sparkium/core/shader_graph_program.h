#pragma once
#include "sparkium/core/core_util.h"

namespace sparkium {

// Backend-neutral description of a JSON shader-node graph.
//
// `ShaderGraphCompiler` (scene_io/json_scene.cpp) emits both the HLSL used by
// the graphics pipelines and this instruction list, from a single traversal of
// the same JSON document. The native CPU/CUDA backends interpret the
// instruction list (pipelines/native/shared/native_graph.h) because they have
// no HLSL compiler; both forms evaluate the same nodes with the same operands
// in the same order.

// Operand encoding: >= 0 selects a register, <= -2 selects
// `constants[-operand - 2]`, -1 marks an unused slot.
constexpr int32_t kShaderGraphOperandNone = -1;
constexpr int kShaderGraphMaxOperands = 6;
// Registers are reused once a value is dead, so this bound is on the peak
// number of simultaneously live node values, not on the graph size.
constexpr int kShaderGraphMaxRegisters = 48;
constexpr int kShaderGraphSurfaceOutputCount = 21;

enum ShaderGraphOp : uint32_t {
  SHADER_GRAPH_OP_COPY = 0,           // value / rgb / passthrough / bump
  SHADER_GRAPH_OP_TEXTURE_COORDINATE,  // sub: 0 uv, 1 normal, 2 object position
  SHADER_GRAPH_OP_IMAGE_TEXTURE,       // sub: 0 color, 1 alpha, 2 sRGB color; data_offset: texture slot
  SHADER_GRAPH_OP_MAPPING,
  SHADER_GRAPH_OP_NOISE_TEXTURE,
  SHADER_GRAPH_OP_VORONOI_TEXTURE,
  SHADER_GRAPH_OP_GRADIENT_TEXTURE,
  SHADER_GRAPH_OP_WAVE_TEXTURE,  // sub: 0 bands X, 1 bands Y, 2 bands Z, 3 diagonal, 4 rings
  SHADER_GRAPH_OP_SKY_TEXTURE,
  SHADER_GRAPH_OP_VERTEX_ATTRIBUTE,  // sub: 0 color, 1 factor, 2 alpha
  SHADER_GRAPH_OP_OBJECT_INFO,       // sub: 0 location, 1 index, 2 random, 3 one
  SHADER_GRAPH_OP_GEOMETRY_INFO,     // sub: 0 position, 1 normal, 2 incoming, 3 backfacing, 4 zero
  SHADER_GRAPH_OP_LIGHT_PATH,        // sub: ShaderGraphLightPathOutput
  SHADER_GRAPH_OP_INVERT,
  SHADER_GRAPH_OP_MIX,   // sub: ShaderGraphBlendType
  SHADER_GRAPH_OP_MATH,  // sub: ShaderGraphMathOp
  SHADER_GRAPH_OP_COLOR_RAMP,
  SHADER_GRAPH_OP_RGB_CURVES,
  SHADER_GRAPH_OP_HUE_SATURATION,
  SHADER_GRAPH_OP_GAMMA,
  SHADER_GRAPH_OP_BRIGHT_CONTRAST,
  SHADER_GRAPH_OP_LAYER_WEIGHT,  // sub: 0 fresnel, 1 facing
  SHADER_GRAPH_OP_NORMAL_MAP,
  SHADER_GRAPH_OP_COMBINE,
  SHADER_GRAPH_OP_SEPARATE,  // sub: component
  SHADER_GRAPH_OP_INVERT_Y,
  SHADER_GRAPH_OP_BRICK_TEXTURE
};

enum ShaderGraphBlendType : uint32_t {
  SHADER_GRAPH_BLEND_MIX = 0,
  SHADER_GRAPH_BLEND_MULTIPLY,
  SHADER_GRAPH_BLEND_ADD,
  SHADER_GRAPH_BLEND_SUBTRACT,
  SHADER_GRAPH_BLEND_DARKEN,
  SHADER_GRAPH_BLEND_LIGHTEN,
  SHADER_GRAPH_BLEND_SCREEN,
  SHADER_GRAPH_BLEND_DIFFERENCE,
  SHADER_GRAPH_BLEND_DIVIDE,
  SHADER_GRAPH_BLEND_OVERLAY
};

enum ShaderGraphMathOp : uint32_t {
  SHADER_GRAPH_MATH_ADD = 0,
  SHADER_GRAPH_MATH_MULTIPLY,
  SHADER_GRAPH_MATH_SUBTRACT,
  SHADER_GRAPH_MATH_DIVIDE,
  SHADER_GRAPH_MATH_POWER,
  SHADER_GRAPH_MATH_MINIMUM,
  SHADER_GRAPH_MATH_MAXIMUM,
  SHADER_GRAPH_MATH_LESS_THAN,
  SHADER_GRAPH_MATH_GREATER_THAN,
  SHADER_GRAPH_MATH_MULTIPLY_ADD,
  SHADER_GRAPH_MATH_ABSOLUTE,
  SHADER_GRAPH_MATH_SINE,
  SHADER_GRAPH_MATH_COSINE,
  SHADER_GRAPH_MATH_FRACT,
  SHADER_GRAPH_MATH_FLOOR
};

enum ShaderGraphLightPathOutput : uint32_t {
  SHADER_GRAPH_LIGHT_PATH_ZERO = 0,
  SHADER_GRAPH_LIGHT_PATH_IS_CAMERA,
  SHADER_GRAPH_LIGHT_PATH_IS_SHADOW,
  SHADER_GRAPH_LIGHT_PATH_IS_REFLECTION,
  SHADER_GRAPH_LIGHT_PATH_IS_TRANSMISSION,
  SHADER_GRAPH_LIGHT_PATH_RAY_LENGTH
};

struct ShaderGraphInstruction {
  uint32_t op{SHADER_GRAPH_OP_COPY};
  uint32_t sub{0};
  int32_t dst{0};
  int32_t operands[kShaderGraphMaxOperands]{kShaderGraphOperandNone, kShaderGraphOperandNone,
                                            kShaderGraphOperandNone, kShaderGraphOperandNone,
                                            kShaderGraphOperandNone, kShaderGraphOperandNone};
  uint32_t data_offset{0};
  uint32_t data_count{0};
};

struct ShaderGraphProgram {
  std::vector<ShaderGraphInstruction> instructions;
  std::vector<glm::vec4> constants;
  std::vector<float> data;
  // Operand-encoded source of each `GraphSurface` field, in the order used by
  // the HLSL emitter's output table; `kShaderGraphOperandNone` keeps the
  // default.
  int32_t outputs[kShaderGraphSurfaceOutputCount];
  int32_t register_count{0};

  ShaderGraphProgram() {
    for (int i = 0; i < kShaderGraphSurfaceOutputCount; ++i)
      outputs[i] = kShaderGraphOperandNone;
  }
};

}  // namespace sparkium
