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
  int primitive_index;
  float3 position;
  float2 tex_coord;
  float3 normal;
  float3 geom_normal;
  float3 tangent;
  float signal;
  int object_id;
  int primitive_id;
  bool front_facing;
};

struct RenderContext {
  float3 origin;
  float3 direction;
  float3 radiance;
  float3 throughput;
  HitRecord hit_record;
  RandomDevice rd;
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
  int geom_data_index;
  int surface_shader_index;
  int surface_data_index;
  int light_data_index;
};

struct LightMetadata {
  int sampler_shader_index;
  int sampler_data_index;
  int geometry_data_index;
  uint power_offset;
};

struct SampleDirectionLightingPayload {
  float3 position; // in
  float3 eval; // out
  float3 omega_in; // out
  float pdf; // out
};
