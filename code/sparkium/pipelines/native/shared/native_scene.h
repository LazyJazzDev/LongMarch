#pragma once

// Device-side scene view. Where the HLSL shading core reaches for global
// resource bindings (`bindings.hlsli`), the ported core takes a `SceneView`
// as its first argument; the byte layouts behind every buffer are identical to
// the ones the graphics pipelines upload.

#include "native_buffer.h"

namespace sparkium::native {

// Mirrors `SOFTWARE_DATA_BUFFER_COUNT` slots of `data_buffers[]`: the shading
// core indexes geometry/material blobs through this array.
struct RenderSettings {
  // Scene settings.
  int samples_per_dispatch;
  int max_bounces;
  int alpha_shadow;
  int _settings_padding;
  float3 background_color;
  // Film info.
  int accumulated_samples;
  float persistence;
  float clamping;
  float max_exposure;
  int view_transform;
  float exposure;
  float gamma;
  float contrast;
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
  uint32_t power_offset;
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
  int primitive_index;
  int object_index;
  bool front_facing;
};

#define RAY_TYPE_CAMERA 0
#define RAY_TYPE_REFLECTION 1
#define RAY_TYPE_TRANSMISSION 2
#define RAY_TYPE_VOLUME 3

struct RandomDevice {
  uint32_t offset;
  uint32_t samp;
  uint32_t seed;
  uint32_t dim;
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
  int bounce;
  int ray_type;
  // Homogeneous random-walk state. A negative object index means that the
  // path is outside a participating subsurface medium.
  int medium_object_index;
  int medium_channel;
  float3 medium_sigma_t;
  float3 medium_albedo;
  float medium_ior;
  float medium_sample_distance;
};

struct GeometryPrimitiveSample {
  float3 position;
  float3 normal;
  float2 tex_coord;
  float pdf;
};

// A texture as uploaded by `Scene::RegisterImage`. SDR images are
// R8G8B8A8_UNORM and live in `sdr_textures`, HDR images are
// R32G32B32A32_SFLOAT and live in `hdr_textures`; exactly one data pointer is
// non-null.
struct DeviceTexture {
  const uint32_t *sdr_data;
  const float *hdr_data;
  int width;
  int height;
};

// Defined by `native_materials.h` / `native_graph.h`; the shading core only
// dereferences them after those headers are complete.
struct NativeMaterial;
struct GraphProgram;

// Flat, pointer-only description of everything the shading core reads.
struct SceneView {
  // Software BVH over instances plus per-instance triangle trees.
  ByteBuffer software_nodes;
  ByteBuffer software_instances;

  // Geometry and material blobs, indexed by `InstanceMetadata`.
  const ByteBuffer *data_buffers;
  uint32_t data_buffer_count;

  ByteBuffer sobol_table;
  ByteBuffer camera_data;
  ByteBuffer instance_metadatas;
  ByteBuffer light_selector_data;
  ByteBuffer light_metadatas;

  const DeviceTexture *sdr_textures;
  uint32_t sdr_texture_count;
  const DeviceTexture *hdr_textures;
  uint32_t hdr_texture_count;

  // The native equivalent of the generated per-material switch: material index
  // as stored in the software instance record selects the sampler to run.
  const NativeMaterial *materials;
  uint32_t material_count;
  const GraphProgram *graph_programs;
  uint32_t graph_program_count;

  RenderSettings settings;

  LM_DEVICE_FUNC ByteBuffer DataBuffer(int index) const {
    if (index < 0 || static_cast<uint32_t>(index) >= data_buffer_count) {
      ByteBuffer empty;
      empty.data = nullptr;
      empty.size = 0;
      return empty;
    }
    return data_buffers[index];
  }

  LM_DEVICE_FUNC InstanceMetadata GetInstanceMetadata(int instance_index) const {
    const uint3 raw = instance_metadatas.Load3(static_cast<uint32_t>(instance_index) * 12);
    InstanceMetadata metadata;
    metadata.geometry_data_index = static_cast<int>(raw.x);
    metadata.material_data_index = static_cast<int>(raw.y);
    metadata.custom_index = static_cast<int>(raw.z);
    return metadata;
  }

  LM_DEVICE_FUNC LightMetadata GetLightMetadata(int light_index) const {
    const uint4 raw = light_metadatas.Load4(static_cast<uint32_t>(light_index) * 16);
    LightMetadata metadata;
    metadata.sampler_shader_index = static_cast<int>(raw.x);
    metadata.sampler_data_index = static_cast<int>(raw.y);
    metadata.custom_index = static_cast<int>(raw.z);
    metadata.power_offset = raw.w;
    return metadata;
  }
};

// Bilinear + repeat fetch, matching the LINEAR/REPEAT sampler that
// `SampleTexture` binds as `samplers[0]`.
LM_DEVICE_FUNC inline float4 FetchTexel(const DeviceTexture &texture, int x, int y) {
  // REPEAT addressing.
  x = x % texture.width;
  if (x < 0)
    x += texture.width;
  y = y % texture.height;
  if (y < 0)
    y += texture.height;
  const int index = y * texture.width + x;
  if (texture.hdr_data) {
    const float *p = texture.hdr_data + index * 4;
    return float4{p[0], p[1], p[2], p[3]};
  }
  const uint32_t packed = texture.sdr_data[index];
  return float4{static_cast<float>(packed & 0xFFu) / 255.0f, static_cast<float>((packed >> 8) & 0xFFu) / 255.0f,
                static_cast<float>((packed >> 16) & 0xFFu) / 255.0f,
                static_cast<float>((packed >> 24) & 0xFFu) / 255.0f};
}

LM_DEVICE_FUNC inline float4 SampleTextureLinear(const DeviceTexture &texture, const float2 &uv) {
  if (texture.width <= 0 || texture.height <= 0)
    return float4{1.0f, 0.0f, 1.0f, 1.0f};
  const float fx = uv.x * static_cast<float>(texture.width) - 0.5f;
  const float fy = uv.y * static_cast<float>(texture.height) - 0.5f;
  const float x0f = ::floorf(fx);
  const float y0f = ::floorf(fy);
  const int x0 = static_cast<int>(x0f);
  const int y0 = static_cast<int>(y0f);
  const float tx = fx - x0f;
  const float ty = fy - y0f;
  const float4 c00 = FetchTexel(texture, x0, y0);
  const float4 c10 = FetchTexel(texture, x0 + 1, y0);
  const float4 c01 = FetchTexel(texture, x0, y0 + 1);
  const float4 c11 = FetchTexel(texture, x0 + 1, y0 + 1);
  return lerp(lerp(c00, c10, tx), lerp(c01, c11, tx), ty);
}

// `SampleTexture` from `bindings.hlsli`: the 0x1000000 bit tags HDR images,
// and the V coordinate is flipped before sampling.
LM_DEVICE_FUNC inline float4 SampleTexture(const SceneView &scene, int texture_index, const float2 &uv) {
  const float2 flipped{uv.x, 1.0f - uv.y};
  if (texture_index & 0x1000000) {
    const uint32_t index = static_cast<uint32_t>(texture_index) & 0xFFFFFFu;
    if (index >= scene.hdr_texture_count)
      return float4{1.0f, 0.0f, 1.0f, 1.0f};
    return SampleTextureLinear(scene.hdr_textures[index], flipped);
  }
  const uint32_t index = static_cast<uint32_t>(texture_index);
  if (index >= scene.sdr_texture_count)
    return float4{1.0f, 0.0f, 1.0f, 1.0f};
  return SampleTextureLinear(scene.sdr_textures[index], flipped);
}

}  // namespace sparkium::native
