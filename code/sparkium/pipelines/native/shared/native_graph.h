#pragma once

// Device-side interpreter for JSON shader-node graphs.
//
// `ShaderGraphCompiler` (scene_io/json_scene.cpp) emits HLSL as a flat list of
// `float4 graph_vN = <expression>;` assignments. The same traversal also emits
// the instruction list interpreted here, one instruction per node, so both the
// HLSL graph and the native graph evaluate the same nodes with the same
// operands in the same order.

#include "native_lighting.h"

namespace sparkium::native {

// Operand encoding: >= 0 selects a register, <= -2 selects
// `constants[-operand - 2]`, -1 marks an unused slot.
#define GRAPH_OPERAND_NONE (-1)
#define GRAPH_MAX_OPERANDS 6
// Must match `kShaderGraphMaxRegisters` (core/shader_graph_program.h).
#define GRAPH_MAX_REGISTERS 48
#define GRAPH_SURFACE_OUTPUT_COUNT 21

enum GraphOp : uint32_t {
  GRAPH_OP_COPY = 0,             // value / rgb / passthrough / bump
  GRAPH_OP_TEXTURE_COORDINATE,   // sub: 0 uv, 1 normal, 2 object position
  GRAPH_OP_IMAGE_TEXTURE,        // sub: 0 color, 1 alpha, 2 sRGB color
  GRAPH_OP_MAPPING,
  GRAPH_OP_NOISE_TEXTURE,
  GRAPH_OP_VORONOI_TEXTURE,
  GRAPH_OP_GRADIENT_TEXTURE,
  GRAPH_OP_WAVE_TEXTURE,  // sub: 0 bands X, 1 bands Y, 2 bands Z, 3 diagonal, 4 rings
  GRAPH_OP_SKY_TEXTURE,
  GRAPH_OP_VERTEX_ATTRIBUTE,  // sub: 0 color, 1 factor, 2 alpha
  GRAPH_OP_OBJECT_INFO,       // sub: 0 location, 1 index, 2 random, 3 one
  GRAPH_OP_GEOMETRY_INFO,     // sub: 0 position, 1 normal, 2 incoming, 3 backfacing, 4 zero
  GRAPH_OP_LIGHT_PATH,        // sub: see GraphLightPathOutput
  GRAPH_OP_INVERT,
  GRAPH_OP_MIX,  // sub: see GraphBlendType
  GRAPH_OP_MATH,  // sub: see GraphMathOp
  GRAPH_OP_COLOR_RAMP,
  GRAPH_OP_RGB_CURVES,
  GRAPH_OP_HUE_SATURATION,
  GRAPH_OP_GAMMA,
  GRAPH_OP_BRIGHT_CONTRAST,
  GRAPH_OP_LAYER_WEIGHT,  // sub: 0 fresnel, 1 facing
  GRAPH_OP_NORMAL_MAP,
  GRAPH_OP_COMBINE,
  GRAPH_OP_SEPARATE,  // sub: component
  GRAPH_OP_INVERT_Y,
  GRAPH_OP_BRICK_TEXTURE
};

enum GraphBlendType : uint32_t {
  GRAPH_BLEND_MIX = 0,
  GRAPH_BLEND_MULTIPLY,
  GRAPH_BLEND_ADD,
  GRAPH_BLEND_SUBTRACT,
  GRAPH_BLEND_DARKEN,
  GRAPH_BLEND_LIGHTEN,
  GRAPH_BLEND_SCREEN,
  GRAPH_BLEND_DIFFERENCE,
  GRAPH_BLEND_DIVIDE,
  GRAPH_BLEND_OVERLAY
};

enum GraphMathOp : uint32_t {
  GRAPH_MATH_ADD = 0,
  GRAPH_MATH_MULTIPLY,
  GRAPH_MATH_SUBTRACT,
  GRAPH_MATH_DIVIDE,
  GRAPH_MATH_POWER,
  GRAPH_MATH_MINIMUM,
  GRAPH_MATH_MAXIMUM,
  GRAPH_MATH_LESS_THAN,
  GRAPH_MATH_GREATER_THAN,
  GRAPH_MATH_MULTIPLY_ADD,
  GRAPH_MATH_ABSOLUTE,
  GRAPH_MATH_SINE,
  GRAPH_MATH_COSINE,
  GRAPH_MATH_FRACT,
  GRAPH_MATH_FLOOR
};

enum GraphLightPathOutput : uint32_t {
  GRAPH_LIGHT_PATH_ZERO = 0,
  GRAPH_LIGHT_PATH_IS_CAMERA,
  GRAPH_LIGHT_PATH_IS_SHADOW,
  GRAPH_LIGHT_PATH_IS_REFLECTION,
  GRAPH_LIGHT_PATH_IS_TRANSMISSION,
  GRAPH_LIGHT_PATH_RAY_LENGTH
};

struct GraphInstruction {
  uint32_t op;
  uint32_t sub;
  int32_t dst;  // Register index, assigned by linear-scan allocation.
  int32_t operands[GRAPH_MAX_OPERANDS];
  uint32_t data_offset;  // Payload window in `GraphProgram::data`.
  uint32_t data_count;
};

// A compiled graph. All pointers address device-resident storage owned by the
// backend's flattened scene.
struct GraphProgram {
  const GraphInstruction *instructions;
  uint32_t instruction_count;
  const float4 *constants;
  const float *data;
  // Operand-encoded source of each `GraphSurface` field, GRAPH_OPERAND_NONE
  // when the graph leaves the default untouched.
  int32_t outputs[GRAPH_SURFACE_OUTPUT_COUNT];
};

// Mirrors `GraphSurface` from material/shader_graph/surface_sampler.hlsli.
struct GraphSurface {
  float3 base_color;
  float metallic;
  float specular;
  float roughness;
  float anisotropic;
  float anisotropic_rotation;
  float sheen;
  float clearcoat;
  float clearcoat_roughness;
  float ior;
  float transmission;
  float transmission_roughness;
  float3 emission;
  float3 normal;
  float opacity;
  float shadow_opacity;
  float thin_walled;
  float subsurface;
  float subsurface_scale;
  float3 subsurface_radius;
  float subsurface_method;
};

// ---------------------------------------------------------------------------
// Graph helper functions, ported from the HLSL prologue.
// ---------------------------------------------------------------------------
LM_DEVICE_FUNC inline float GraphHash(float3 p) {
  p = frac(p * 0.3183099f + 0.1f);
  p *= 17.0f;
  return frac(p.x * p.y * p.z * (p.x + p.y + p.z));
}

LM_DEVICE_FUNC inline float GraphNoise(const float3 &p) {
  const float3 i = glm::floor(p);
  float3 f = frac(p);
  f = f * f * (3.0f - 2.0f * f);
  const float n000 = GraphHash(i), n100 = GraphHash(i + float3{1, 0, 0});
  const float n010 = GraphHash(i + float3{0, 1, 0}), n110 = GraphHash(i + float3{1, 1, 0});
  const float n001 = GraphHash(i + float3{0, 0, 1}), n101 = GraphHash(i + float3{1, 0, 1});
  const float n011 = GraphHash(i + float3{0, 1, 1}), n111 = GraphHash(i + float3{1, 1, 1});
  return lerp(lerp(lerp(n000, n100, f.x), lerp(n010, n110, f.x), f.y),
              lerp(lerp(n001, n101, f.x), lerp(n011, n111, f.x), f.y), f.z);
}

LM_DEVICE_FUNC inline float3 GraphSrgbToLinear(const float3 &color) {
  const float3 lower = color / 12.92f;
  const float3 upper{::powf((color.x + 0.055f) / 1.055f, 2.4f), ::powf((color.y + 0.055f) / 1.055f, 2.4f),
                     ::powf((color.z + 0.055f) / 1.055f, 2.4f)};
  return lerp(lower, upper, step(make_float3(0.04045f), color));
}

LM_DEVICE_FUNC inline float GraphVoronoi(const float3 &p) {
  const float3 cell = glm::floor(p);
  const float3 local = frac(p);
  float distance = 1e9f;
  for (int z = -1; z <= 1; ++z)
    for (int y = -1; y <= 1; ++y)
      for (int x = -1; x <= 1; ++x) {
        const float3 o{static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)};
        const float3 sample_point = o + GraphHash(cell + o + float3{0, 0, 0}) * float3{0.73f, 0.91f, 0.57f};
        distance = ::fminf(distance, glm::length(sample_point - local));
      }
  return distance;
}

LM_DEVICE_FUNC inline float3 GraphRotateXYZ(float3 p, const float3 &r) {
  const float3 s{::sinf(r.x), ::sinf(r.y), ::sinf(r.z)};
  const float3 c{::cosf(r.x), ::cosf(r.y), ::cosf(r.z)};
  p = float3{p.x, c.x * p.y - s.x * p.z, s.x * p.y + c.x * p.z};
  p = float3{c.y * p.x + s.y * p.z, p.y, -s.y * p.x + c.y * p.z};
  return float3{c.z * p.x - s.z * p.y, s.z * p.x + c.z * p.y, p.z};
}

LM_DEVICE_FUNC inline float3 GraphRgbToHsv(const float3 &c) {
  const float4 K{0.0f, -1.0f / 3.0f, 2.0f / 3.0f, -1.0f};
  const float4 p = lerp(float4{c.z, c.y, K.w, K.z}, float4{c.y, c.z, K.x, K.y}, step(c.z, c.y));
  const float4 q = lerp(float4{p.x, p.y, p.w, c.x}, float4{c.x, p.y, p.z, p.x}, step(p.x, c.x));
  const float d = q.x - ::fminf(q.w, q.y), e = 1.0e-10f;
  return float3{::fabsf(q.z + (q.w - q.y) / (6.0f * d + e)), d / (q.x + e), q.x};
}

LM_DEVICE_FUNC inline float3 GraphHsvToRgb(const float3 &c) {
  const float3 p = glm::abs(frac(float3{c.x, c.x, c.x} + float3{0.0f, 2.0f / 3.0f, 1.0f / 3.0f}) * 6.0f - 3.0f);
  return c.z * lerp(float3{1, 1, 1}, saturate(p - 1.0f), c.y);
}

// ---------------------------------------------------------------------------
// Interpreter
// ---------------------------------------------------------------------------
struct GraphRegisters {
  float4 values[GRAPH_MAX_REGISTERS];
};

LM_DEVICE_FUNC inline float4 GraphOperand(const GraphProgram &program, const GraphRegisters &registers, int32_t slot) {
  if (slot >= 0)
    return registers.values[slot];
  if (slot == GRAPH_OPERAND_NONE)
    return float4{0, 0, 0, 0};
  return program.constants[-slot - 2];
}

LM_DEVICE_FUNC inline float4 GraphSplat(float v) {
  return float4{v, v, v, v};
}

LM_DEVICE_FUNC inline float3 GraphXYZ(const float4 &v) {
  return float3{v.x, v.y, v.z};
}

LM_DEVICE_FUNC inline float4 EvaluateGraphInstruction(const SceneView &scene,
                                                      const GraphProgram &program,
                                                      const GraphInstruction &instruction,
                                                      const GraphRegisters &registers,
                                                      const ByteBuffer &material_data,
                                                      const HitRecord &hit_record,
                                                      const float3 &view_direction,
                                                      int bounce,
                                                      int ray_type,
                                                      bool is_shadow_ray) {
#define OPERAND(index) GraphOperand(program, registers, instruction.operands[index])
  switch (instruction.op) {
    case GRAPH_OP_COPY:
      return OPERAND(0);
    case GRAPH_OP_TEXTURE_COORDINATE:
      if (instruction.sub == 0)
        return float4{hit_record.tex_coord.x, hit_record.tex_coord.y, 0, 0};
      if (instruction.sub == 1)
        return float4{hit_record.normal, 0};
      return float4{hit_record.object_position, 0};
    case GRAPH_OP_IMAGE_TEXTURE: {
      const uint32_t slot = instruction.data_offset;
      const int texture_index = static_cast<int>(material_data.Load(12 + slot * 4));
      const float4 vector = OPERAND(0);
      const float4 texel = SampleTexture(scene, texture_index, float2{vector.x, vector.y});
      if (instruction.sub == 1)
        return GraphSplat(texel.w);
      if (instruction.sub == 2)
        return float4{GraphSrgbToLinear(GraphXYZ(texel)), texel.w};
      return texel;
    }
    case GRAPH_OP_MAPPING: {
      const float4 vector = OPERAND(0), location = OPERAND(1), rotation = OPERAND(2), scale = OPERAND(3);
      return float4{GraphRotateXYZ(GraphXYZ(vector) * GraphXYZ(scale), GraphXYZ(rotation)) + GraphXYZ(location), 0};
    }
    case GRAPH_OP_NOISE_TEXTURE: {
      const float4 vector = OPERAND(0), scale = OPERAND(1);
      return GraphSplat(GraphNoise(GraphXYZ(vector) * scale.x));
    }
    case GRAPH_OP_VORONOI_TEXTURE: {
      const float4 vector = OPERAND(0), scale = OPERAND(1);
      return GraphSplat(GraphVoronoi(GraphXYZ(vector) * scale.x));
    }
    case GRAPH_OP_GRADIENT_TEXTURE:
      return GraphSplat(OPERAND(0).x);
    case GRAPH_OP_WAVE_TEXTURE: {
      const float4 vector = OPERAND(0), scale = OPERAND(1), distortion = OPERAND(2), phase = OPERAND(3);
      const float3 p = GraphXYZ(vector) * scale.x;
      float coordinate;
      switch (instruction.sub) {
        case 4:
          coordinate = glm::length(p);
          break;
        case 1:
          coordinate = p.y;
          break;
        case 2:
          coordinate = p.z;
          break;
        case 3:
          coordinate = (p.x + p.y + p.z) * 0.577350269f;
          break;
        default:
          coordinate = p.x;
          break;
      }
      coordinate = coordinate + distortion.x * GraphNoise(p) + phase.x;
      return GraphSplat(0.5f + 0.5f * ::sinf(coordinate * 6.283185307f));
    }
    case GRAPH_OP_SKY_TEXTURE: {
      const float4 vector = OPERAND(0), sun = OPERAND(1);
      const float3 direction = glm::normalize(GraphXYZ(vector));
      const float3 sky = lerp(float3{0.08f, 0.16f, 0.35f}, float3{0.65f, 0.78f, 1.0f},
                              saturatef(direction.z * 0.5f + 0.5f)) +
                         ::powf(saturatef(glm::dot(direction, glm::normalize(GraphXYZ(sun)))), 512.0f) *
                             float3{8, 6, 3};
      return float4{sky, 1};
    }
    case GRAPH_OP_VERTEX_ATTRIBUTE:
      if (instruction.sub == 2)
        return float4{1, 1, 1, 1};
      if (instruction.sub == 1)
        return GraphSplat(hit_record.color.x);
      return float4{hit_record.color, 1};
    case GRAPH_OP_OBJECT_INFO:
      if (instruction.sub == 0)
        return float4{hit_record.object_origin, 0};
      if (instruction.sub == 1)
        return GraphSplat(static_cast<float>(hit_record.object_index));
      if (instruction.sub == 2)
        return GraphSplat(GraphHash(float3{static_cast<float>(hit_record.object_index), 17.0f, 31.0f}));
      return float4{1, 1, 1, 1};
    case GRAPH_OP_GEOMETRY_INFO:
      if (instruction.sub == 0)
        return float4{hit_record.position, 1};
      if (instruction.sub == 1)
        return float4{hit_record.normal, 0};
      if (instruction.sub == 2)
        return float4{-view_direction, 0};
      if (instruction.sub == 3)
        return GraphSplat(hit_record.front_facing ? 0.0f : 1.0f);
      return float4{0, 0, 0, 0};
    case GRAPH_OP_LIGHT_PATH:
      switch (instruction.sub) {
        case GRAPH_LIGHT_PATH_IS_CAMERA:
          return GraphSplat(!is_shadow_ray && ray_type == RAY_TYPE_CAMERA ? 1.0f : 0.0f);
        case GRAPH_LIGHT_PATH_IS_SHADOW:
          return GraphSplat(is_shadow_ray ? 1.0f : 0.0f);
        case GRAPH_LIGHT_PATH_IS_REFLECTION:
          return GraphSplat(!is_shadow_ray && ray_type == RAY_TYPE_REFLECTION ? 1.0f : 0.0f);
        case GRAPH_LIGHT_PATH_IS_TRANSMISSION:
          return GraphSplat(!is_shadow_ray && ray_type == RAY_TYPE_TRANSMISSION ? 1.0f : 0.0f);
        case GRAPH_LIGHT_PATH_RAY_LENGTH:
          return GraphSplat(hit_record.t);
        default:
          return float4{0, 0, 0, 0};
      }
    case GRAPH_OP_INVERT: {
      const float4 color = OPERAND(0), factor = OPERAND(1);
      return lerp(color, 1.0f - color, factor.x);
    }
    case GRAPH_OP_MIX: {
      const float4 factor = OPERAND(0), a = OPERAND(1), b = OPERAND(2);
      const float t = saturatef(factor.x);
      switch (instruction.sub) {
        case GRAPH_BLEND_MULTIPLY:
          return lerp(a, a * b, t);
        case GRAPH_BLEND_ADD:
          return a + b * factor.x;
        case GRAPH_BLEND_SUBTRACT:
          return a - b * factor.x;
        case GRAPH_BLEND_DARKEN:
          return lerp(a, glm::min(a, b), t);
        case GRAPH_BLEND_LIGHTEN:
          return lerp(a, glm::max(a, b), t);
        case GRAPH_BLEND_SCREEN:
          return lerp(a, 1.0f - (1.0f - a) * (1.0f - b), t);
        case GRAPH_BLEND_DIFFERENCE:
          return lerp(a, glm::abs(a - b), t);
        case GRAPH_BLEND_DIVIDE:
          return lerp(a, a / glm::max(glm::abs(b), float4{1e-8f, 1e-8f, 1e-8f, 1e-8f}), t);
        case GRAPH_BLEND_OVERLAY: {
          const float4 overlay = lerp(2.0f * a * b, 1.0f - 2.0f * (1.0f - a) * (1.0f - b), step(float4{0.5f, 0.5f, 0.5f, 0.5f}, a));
          return lerp(a, overlay, t);
        }
        default:
          return lerp(a, b, t);
      }
    }
    case GRAPH_OP_MATH: {
      const float x = OPERAND(0).x, y = OPERAND(1).x, z = OPERAND(2).x;
      float scalar;
      switch (instruction.sub) {
        case GRAPH_MATH_MULTIPLY:
          scalar = x * y;
          break;
        case GRAPH_MATH_SUBTRACT:
          scalar = x - y;
          break;
        case GRAPH_MATH_DIVIDE:
          scalar = x / ::fmaxf(::fabsf(y), 1e-8f);
          break;
        case GRAPH_MATH_POWER:
          scalar = ::powf(::fmaxf(x, 0.0f), y);
          break;
        case GRAPH_MATH_MINIMUM:
          scalar = ::fminf(x, y);
          break;
        case GRAPH_MATH_MAXIMUM:
          scalar = ::fmaxf(x, y);
          break;
        case GRAPH_MATH_LESS_THAN:
          scalar = x < y ? 1.0f : 0.0f;
          break;
        case GRAPH_MATH_GREATER_THAN:
          scalar = x > y ? 1.0f : 0.0f;
          break;
        case GRAPH_MATH_MULTIPLY_ADD:
          scalar = x * y + z;
          break;
        case GRAPH_MATH_ABSOLUTE:
          scalar = ::fabsf(x);
          break;
        case GRAPH_MATH_SINE:
          scalar = ::sinf(x);
          break;
        case GRAPH_MATH_COSINE:
          scalar = ::cosf(x);
          break;
        case GRAPH_MATH_FRACT:
          scalar = frac(x);
          break;
        case GRAPH_MATH_FLOOR:
          scalar = ::floorf(x);
          break;
        default:
          scalar = x + y;
          break;
      }
      return GraphSplat(scalar);
    }
    case GRAPH_OP_COLOR_RAMP: {
      const float4 factor = OPERAND(0);
      // Payload: (position, span, r, g, b, a) per element, where span is the
      // interval to the previous element as baked into the HLSL expression.
      const uint32_t count = instruction.sub;
      if (count == 0)
        return factor;
      const float *elements = program.data + instruction.data_offset;
      float4 result{elements[2], elements[3], elements[4], elements[5]};
      for (uint32_t i = 1; i < count; ++i) {
        const float *element = elements + i * 6;
        const float t = saturatef((factor.x - element[-6]) / element[1]);
        result = lerp(result, float4{element[2], element[3], element[4], element[5]}, t);
      }
      return result;
    }
    case GRAPH_OP_RGB_CURVES: {
      const float4 color = OPERAND(0), factor = OPERAND(1);
      // Payload: three self-describing channel blocks laid out back to back,
      // each `count, scale, count samples, count positions`. The positions and
      // the scale are the constants the HLSL emitter bakes into the lerp
      // chain, so both forms evaluate the same arithmetic.
      const float *data = program.data + instruction.data_offset;
      float channels[3];
      for (int channel = 0; channel < 3; ++channel) {
        const uint32_t count = static_cast<uint32_t>(data[0]);
        const float scale = data[1];
        const float *curve_samples = data + 2;
        const float *positions = curve_samples + count;
        const float x = saturatef(channel == 0 ? color.x : (channel == 1 ? color.y : color.z));
        float curve = curve_samples[0];
        for (uint32_t i = 1; i < count; ++i) {
          const float t = saturatef((x - positions[i - 1]) * scale);
          curve = lerp(curve, curve_samples[i], t);
        }
        channels[channel] = curve;
        data = positions + count;
      }
      const float4 curved{channels[0], channels[1], channels[2], color.w};
      return lerp(color, curved, saturatef(factor.x));
    }
    case GRAPH_OP_HUE_SATURATION: {
      const float4 color = OPERAND(0), factor = OPERAND(1), hue = OPERAND(2), saturation = OPERAND(3),
                   value = OPERAND(4);
      const float3 hsv = GraphRgbToHsv(GraphXYZ(color));
      const float3 adjusted = GraphHsvToRgb(
          float3{frac(hsv.x + hue.x - 0.5f), ::fmaxf(0.0f, hsv.y * saturation.x), hsv.z * value.x});
      return float4{lerp(GraphXYZ(color), adjusted, saturatef(factor.x)), color.w};
    }
    case GRAPH_OP_GAMMA: {
      const float4 color = OPERAND(0), gamma = OPERAND(1);
      const float3 base = glm::max(GraphXYZ(color), make_float3(0.0f));
      const float exponent = ::fmaxf(gamma.x, 1e-6f);
      return float4{::powf(base.x, exponent), ::powf(base.y, exponent), ::powf(base.z, exponent), color.w};
    }
    case GRAPH_OP_BRIGHT_CONTRAST: {
      const float4 color = OPERAND(0), brightness = OPERAND(1), contrast = OPERAND(2);
      return float4{GraphXYZ(color) * (1.0f + contrast.x) + make_float3(brightness.x), color.w};
    }
    case GRAPH_OP_LAYER_WEIGHT: {
      const float4 blend = OPERAND(0);
      const float cosine = glm::dot(glm::normalize(hit_record.normal), glm::normalize(view_direction));
      if (instruction.sub == 1)
        return GraphSplat(1.0f - ::fabsf(cosine));
      return GraphSplat(::powf(1.0f - saturatef(::fabsf(cosine)), ::fmaxf(0.01f, blend.x * 5.0f)));
    }
    case GRAPH_OP_NORMAL_MAP: {
      const float4 color = OPERAND(0), strength = OPERAND(1);
      const float3 tangent_normal =
          glm::normalize(lerp(float3{0, 0, 1}, GraphXYZ(color) * 2.0f - 1.0f, saturatef(strength.x)));
      const float3 mapped_normal =
          glm::normalize(mul_rows(tangent_normal, hit_record.tangent,
                                  glm::cross(hit_record.normal, hit_record.tangent) * hit_record.signal,
                                  hit_record.normal));
      return float4{::fabsf(hit_record.signal) > 0.5f ? mapped_normal : hit_record.normal, 0};
    }
    case GRAPH_OP_COMBINE:
      return float4{OPERAND(0).x, OPERAND(1).x, OPERAND(2).x, 1.0f};
    case GRAPH_OP_SEPARATE: {
      const float4 color = OPERAND(0);
      return GraphSplat(instruction.sub == 1 ? color.y : (instruction.sub == 2 ? color.z : color.x));
    }
    case GRAPH_OP_INVERT_Y: {
      const float4 vector = OPERAND(0);
      return float4{vector.x, 1.0f - vector.y, vector.z, vector.w};
    }
    case GRAPH_OP_BRICK_TEXTURE: {
      const float4 p = OPERAND(0), scale = OPERAND(1), mortar_size = OPERAND(2), c1 = OPERAND(3), c2 = OPERAND(4),
                   mortar = OPERAND(5);
      const float2 q = frac(float2{p.x, p.y} * scale.x);
      const bool edge = ::fminf(::fminf(q.x, q.y), ::fminf(1.0f - q.x, 1.0f - q.y)) < mortar_size.x;
      if (edge)
        return mortar;
      const float parity = ::fmodf(::floorf(p.x * scale.x) + ::floorf(p.y * scale.x), 2.0f);
      return lerp(c1, c2, parity);
    }
    default:
      return float4{0, 0, 0, 0};
  }
#undef OPERAND
}

LM_DEVICE_FUNC inline GraphSurface EvaluateShaderGraph(const SceneView &scene,
                                                       const GraphProgram &program,
                                                       const ByteBuffer &material_data,
                                                       const HitRecord &hit_record,
                                                       const float3 &view_direction,
                                                       int bounce,
                                                       int ray_type,
                                                       bool is_shadow_ray) {
  GraphSurface surface;
  surface.base_color = float3{0.8f, 0.8f, 0.8f};
  surface.metallic = 0.0f;
  surface.specular = 0.5f;
  surface.roughness = 0.5f;
  surface.anisotropic = 0.0f;
  surface.anisotropic_rotation = 0.0f;
  surface.sheen = 0.0f;
  surface.clearcoat = 0.0f;
  surface.clearcoat_roughness = 0.03f;
  surface.ior = 1.45f;
  surface.transmission = 0.0f;
  surface.transmission_roughness = 0.0f;
  surface.emission = float3{0, 0, 0};
  surface.normal = hit_record.normal;
  surface.opacity = 1.0f;
  surface.shadow_opacity = -1.0f;
  surface.thin_walled = 0.0f;
  surface.subsurface = 0.0f;
  surface.subsurface_scale = 0.0f;
  surface.subsurface_radius = float3{1.0f, 0.2f, 0.1f};
  surface.subsurface_method = 0.0f;

  GraphRegisters registers;
  for (uint32_t i = 0; i < program.instruction_count; ++i) {
    const GraphInstruction &instruction = program.instructions[i];
    const float4 value = EvaluateGraphInstruction(scene, program, instruction, registers, material_data, hit_record,
                                                  view_direction, bounce, ray_type, is_shadow_ray);
    registers.values[instruction.dst] = value;
  }

  // Assignment order follows the HLSL emitter's `outputs[]` table.
  for (int i = 0; i < GRAPH_SURFACE_OUTPUT_COUNT; ++i) {
    const int32_t slot = program.outputs[i];
    if (slot == GRAPH_OPERAND_NONE)
      continue;
    const float4 value = GraphOperand(program, registers, slot);
    switch (i) {
      case 0:
        surface.base_color = GraphXYZ(value);
        break;
      case 1:
        surface.metallic = value.x;
        break;
      case 2:
        surface.specular = value.x;
        break;
      case 3:
        surface.roughness = value.x;
        break;
      case 4:
        surface.anisotropic = value.x;
        break;
      case 5:
        surface.anisotropic_rotation = value.x;
        break;
      case 6:
        surface.sheen = value.x;
        break;
      case 7:
        surface.clearcoat = value.x;
        break;
      case 8:
        surface.clearcoat_roughness = value.x;
        break;
      case 9:
        surface.ior = value.x;
        break;
      case 10:
        surface.transmission = value.x;
        break;
      case 11:
        surface.transmission_roughness = value.x;
        break;
      case 12:
        surface.emission = GraphXYZ(value);
        break;
      case 13:
        surface.normal = GraphXYZ(value);
        break;
      case 14:
        surface.opacity = value.x;
        break;
      case 15:
        surface.shadow_opacity = value.x;
        break;
      case 16:
        surface.thin_walled = value.x;
        break;
      case 17:
        surface.subsurface = value.x;
        break;
      case 18:
        surface.subsurface_scale = value.x;
        break;
      case 19:
        surface.subsurface_radius = GraphXYZ(value);
        break;
      case 20:
        surface.subsurface_method = value.x;
        break;
      default:
        break;
    }
  }
  return surface;
}

}  // namespace sparkium::native
