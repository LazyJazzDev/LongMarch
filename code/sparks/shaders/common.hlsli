#pragma once
#include "constants.hlsli"

struct RandomDevice {
  uint offset;
  uint samp;
  uint seed;
  uint dim;
} random_device;

struct CameraData {
 float4x4 world_to_camera;
 float4x4 camera_to_world;
 float2 scale;
};

struct HitRecord {
  float t;
  float3 position;
  float2 tex_coord;
  float3 normal;
  float3 geom_normal;
  float3 tangent;
  float signal;
  float pdf;
  int primitive_index;
  int object_index;
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
};

struct ShadowRayPayload {
  float shadow;
};

struct RayGenPayload {
  float2 uv;
  float3 origin;
  float3 direction;
};

struct RenderSettings {
  // Scene Settings
  int samples_per_dispatch;
  int max_bounces;
  bool alpha_shadow;
  // Film Info
  int accumulated_samples;
  float persistence;
  float clamping;
  float max_exposure;
};

struct InstanceMetadata {
  int geometry_data_index;
  int material_data_index;
  int custom_index;
};

struct LightMetadata {
  int sampler_shader_index;
  int sampler_data_index;
  int custom_index;
  uint power_offset;
};

struct SampleDirectLightingPayload {
  uint4 low;
  uint4 high;
};

struct GeometryPrimitiveSample {
  float3 position;
  float3 normal;
  float2 tex_coord;
  float pdf;
};
