#pragma once
// Portable mirror of code/sparkium/shaders/common.hlsli plus the typed scene
// description consumed by the offline (CPU and CUDA) backends.
//
// The Vulkan/D3D12/Metal backends keep their byte-address buffers and compile
// the HLSL in code/sparkium/shaders; the offline backends describe the same
// scene with typed arrays so a single transcription of the shading core can run
// both on the host and inside CUDA kernels.

#include <stdint.h>

#include "sparkium/backends/core/constants.h"
#include "sparkium/backends/core/hlsl_math.h"

namespace sparkium::backends {

using device::float2;
using device::float3;
using device::float4;
using device::float3x3;
using device::float4x4;
using device::uint2;
using device::uint3;
using device::uint4;

#define SPARKIUM_RAY_TYPE_CAMERA 0
#define SPARKIUM_RAY_TYPE_REFLECTION 1
#define SPARKIUM_RAY_TYPE_TRANSMISSION 2
#define SPARKIUM_RAY_TYPE_VOLUME 3

// ---------------------------------------------------------------------------
// Transforms
// ---------------------------------------------------------------------------

// Column-major affine transform: the exact layout the graphics backends upload
// for `glm::mat4x3` instances, so `mul(matrix, float4(p, 1))` in the HLSL maps
// onto `transform_point`.
struct Mat4x3 {
  float3 c0, c1, c2, c3;
};

SPARKIUM_HD inline float3 transform_point(const Mat4x3 &m, const float3 &p) {
  return m.c0 * p.x + m.c1 * p.y + m.c2 * p.z + m.c3;
}

SPARKIUM_HD inline float3 transform_vector(const Mat4x3 &m, const float3 &v) {
  return m.c0 * v.x + m.c1 * v.y + m.c2 * v.z;
}

// `mul(float3x4, float4)` uses every component (the affine case).
SPARKIUM_HD inline float3 mul(const Mat4x3 &m, const float4 &v) {
  return m.c0 * v.x + m.c1 * v.y + m.c2 * v.z + m.c3 * v.w;
}

// `mul(float4x3, float3)` (and HLSL's implicit float4 -> float3 truncation) is a
// pure direction transform without translation.
SPARKIUM_HD inline float3 mul(const Mat4x3 &m, const float3 &v) {
  return transform_vector(m, v);
}

// ---------------------------------------------------------------------------
// Shader mirror structs
// ---------------------------------------------------------------------------

struct RandomDevice {
  uint32_t offset;
  uint32_t samp;
  uint32_t seed;
  uint32_t dim;
};

struct HitRecord {
  float t;
  float3 position;
  float3 object_position;
  float3 object_origin;
  float2 tex_coord;
  float3 color;
  float3 normal;
  float3 geom_normal;
  float3 tangent;
  float signal;
  float pdf;
  int32_t primitive_index;
  int32_t object_index;
  bool front_facing;
};

struct RenderContext {
  float3 origin;
  float3 direction;
  float3 radiance;
  float3 throughput;
  RandomDevice rd;
  float bsdf_pdf;
  float3 shadow_eval;
  float3 shadow_dir;
  float shadow_length;
  int32_t bounce;
  int32_t ray_type;
  // Kept for parity with common.hlsli; the offline backends always render
  // outside a participating medium because only shader-graph materials can
  // start a subsurface random walk, and those materials are not supported.
  int32_t medium_object_index;
};

// Mirrors Scene::Settings::RayTracing plus the subset of Film::Info the shaders
// read through render_settings.
struct RenderSettings {
  int32_t samples_per_dispatch;
  int32_t max_bounces;
  int32_t alpha_shadow;
  int32_t accumulated_samples;
  float3 background_color;
  float persistence;
  float clamping;
  float max_exposure;
  int32_t view_transform;
  float exposure;
  float gamma;
  float contrast;
};

struct ShadowRayPayload {
  float shadow;
};

struct RayGenPayload {
  float2 uv;
  float2 lens_sample;
  float3 origin;
  float3 direction;
};

struct GeometryPrimitiveSample {
  float3 position;
  float3 normal;
  float2 tex_coord;
  float pdf;
};

// The HLSL light samplers communicate through an asuint() payload whose
// low/high/extra members are read as inputs and overwritten as outputs. The
// portable core keeps those roles as named fields so the samplers read the same.
struct SampleDirectLightingPayload {
  // Inputs (payload.low.xyz / payload.high.xyz / payload.extra.xyz).
  float3 position;
  float3 sample;
  float3 normal;
  int32_t custom_index;
  // Outputs.
  float3 eval;  // payload.low.xyz
  float shadow_length;
  float3 omega_in;  // payload.high.xyz
  float pdf;        // payload.high.w
};

// ---------------------------------------------------------------------------
// Scene description
// ---------------------------------------------------------------------------

enum MaterialKind {
  MATERIAL_KIND_LAMBERTIAN = 0,
  MATERIAL_KIND_SPECULAR = 1,
  MATERIAL_KIND_LIGHT = 2,
  MATERIAL_KIND_PRINCIPLED = 3,
};

enum LightSamplerIndex {
  LIGHT_SAMPLER_POINT = 0x1000000,
  LIGHT_SAMPLER_MESH_LIGHT = 0x1000001,
  LIGHT_SAMPLER_MESH_LAMBERTIAN = 0x1000002,
  LIGHT_SAMPLER_MESH_PRINCIPLED = 0x1000003,
  LIGHT_SAMPLER_MESH_SHADER_GRAPH = 0x1000004,
  LIGHT_SAMPLER_MESH_SPECULAR = 0x1000005,
};

// Texture slots of the principled material, in the order
// material/principled/sampler.hlsl streams them.
enum PrincipledTextureSlot {
  PRINCIPLED_TEXTURE_NORMAL = 0,
  PRINCIPLED_TEXTURE_BASE_COLOR = 1,
  PRINCIPLED_TEXTURE_METALLIC = 2,
  PRINCIPLED_TEXTURE_SPECULAR = 3,
  PRINCIPLED_TEXTURE_ROUGHNESS = 4,
  PRINCIPLED_TEXTURE_ANISOTROPIC = 5,
  PRINCIPLED_TEXTURE_ANISOTROPIC_ROTATION = 6,
  PRINCIPLED_TEXTURE_EMISSION = 7,
  PRINCIPLED_TEXTURE_SLOT_COUNT = 8,
};

struct PrincipledParams {
  float3 base_color;
  float3 subsurface_color;
  float subsurface;
  float3 subsurface_radius;
  float metallic;
  float specular;
  float specular_tint;
  float roughness;
  float anisotropic;
  float anisotropic_rotation;
  float sheen;
  float sheen_tint;
  float clearcoat;
  float clearcoat_roughness;
  float ior;
  float transmission;
  float transmission_roughness;
  float3 emission_color;
  float emission_strength;
  int32_t texture_index[PRINCIPLED_TEXTURE_SLOT_COUNT];
  float normal_y_signal;
};

struct MaterialData {
  int32_t kind;
  // lambertian / specular
  float3 base_color;
  // lambertian emission
  float3 emission;
  // light material
  int32_t two_sided;
  int32_t block_ray;
  int32_t camera_visible;
  float falloff_distance;
  PrincipledParams principled;
};

// One mesh: a byte blob identical to the graphics backends' mesh buffer
// (GeometryHeader followed by the vertex arrays) plus its BVH root node.
struct MeshRange {
  uint32_t offset;           // byte offset into DeviceScene::mesh_data
  uint32_t root;             // BVH root node index
  uint32_t primitive_count;  // triangle count
};

struct InstanceData {
  Mat4x3 object_to_world;
  Mat4x3 world_to_object;
  uint32_t mesh;
  uint32_t material;
  int32_t light;  // light index used for emitter MIS, -1 when the mesh emits nothing
};

struct LightData {
  int32_t sampler_shader_index;
  int32_t custom_index;  // instance index for mesh lights, -1 for point lights
  float power;           // value inserted into the light selection CDF
  // Point lights (mirrors LightPoint::SamplerPreprocess).
  float3 light_position;
  float3 light_power;
  float radius;
  int32_t soft_falloff;
  // Mesh lights (mirrors LightGeometryMaterial sampler data).
  uint32_t primitive_cdf_offset;
  uint32_t primitive_count;
  Mat4x3 mesh_transform;
};

struct TextureData {
  uint32_t offset;  // float index into DeviceScene::texture_pixels (RGBA)
  uint32_t width;
  uint32_t height;
  int32_t hdr;
};

// BVH node, byte-compatible with software/layout.hlsli `SoftwareNode`.
struct SoftwareNode {
  float3 lo;
  uint32_t first;
  float3 hi;
  uint32_t second;
};

static const uint32_t SPARKIUM_SOFTWARE_INVALID = 0xffffffffu;

// A flattened, pointer-based description of the scene. The same struct is used
// on the host (CPU backend) and in device memory (CUDA backend).
struct DeviceScene {
  const uint8_t *mesh_data;
  const MeshRange *meshes;
  uint32_t num_meshes;
  const SoftwareNode *mesh_nodes;
  const SoftwareNode *instance_nodes;
  const InstanceData *instances;
  uint32_t num_instances;
  const MaterialData *materials;
  uint32_t num_materials;
  const LightData *lights;
  uint32_t num_lights;
  const float *light_power_cdf;      // inclusive prefix sums over lights
  const float *primitive_power_cdf;  // concatenated per mesh light
  const TextureData *textures;
  uint32_t num_textures;
  const float *texture_pixels;  // RGBA, four floats per texel
  const uint32_t *sobol_table;
  // Camera, mirroring raytracing::CameraData.
  float2 camera_scale;
  float aperture_radius;
  float focus_distance;
  int32_t aperture_blades;
  float aperture_rotation;
  float aperture_ratio;
  float4x4 camera_to_world;
};

}  // namespace sparkium::backends
