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
  int material_buffer_index;
  RandomDevice rd;
};

struct RayGenPayload {
  float2 uv;
  float3 origin;
  float3 direction;
};

struct MaterialRegistration {
  int shader_index;
  int buffer_index;
};

struct SceneSettings {
  int samples_per_dispatch;
  int max_bounces;
};

struct LightRegistration {
  int shader_index;
  int buffer_index;
};

struct EntityInfo {
  MaterialRegistration material_registration;
  LightRegistration light_registration;
};
