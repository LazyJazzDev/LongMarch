#pragma once
#include "constants.hlsli"
#include "random.hlsli"

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
  HitRecord hit_record;
  RandomDevice rd;
  float bsdf_pdf;
};

struct RayGenPayload {
  float2 uv;
  float3 origin;
  float3 direction;
};

struct SceneSettings {
  int samples_per_dispatch;
  int max_bounces;
};

struct InstanceMetadata {
  int geometry_data_index;
  int surface_shader_index;
  int surface_data_index;
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
