#pragma once
#include "common.hlsli"
#include "buffer_helper.hlsli"

// Each backend binds the same set of slots in its own way: the two GPU paths
// declare device resources through register(...), while the CPU path binds
// host-side views of the same data (see
// pipelines/raytracing/cpu/shaders/hlsl_cpu_bindings.h). Only the declarations
// differ; the slot index macros below are shared by all of them.
#ifndef SPARKIUM_CPU_SHADER
RWTexture2D<float4> accumulated_color : register(u0, space0);
RWTexture2D<float> accumulated_samples : register(u0, space1);
ConstantBuffer<RenderSettings> render_settings : register(b0, space3);
#endif
#define SOBOL_TABLE
#ifdef SPARKIUM_SOFTWARE_RT
#ifndef SPARKIUM_CPU_SHADER
#ifdef SPARKIUM_RAY_QUERY
RaytracingAccelerationStructure query_scene : register(t0, space2);
#else
ByteAddressBuffer software_nodes : register(t0, space2);
#endif
ByteAddressBuffer data_buffers[] : register(t0, space4);
Texture2D<float4> sdr_textures[] : register(t0, space5);
Texture2D<float4> hdr_textures[] : register(t0, space6);
SamplerState samplers[] : register(s0, space7);
#else
// The CPU backend is compiled once for every scene, so the number of scene
// buffers is not known at compile time; the host supplies it.
#define SOFTWARE_DATA_BUFFER_COUNT sparkium_cpu_shaders::software_data_buffer_count
#endif
#define sobol_table data_buffers[SOFTWARE_DATA_BUFFER_COUNT]
#define camera_data data_buffers[SOFTWARE_DATA_BUFFER_COUNT + 1]
#define instance_metadatas data_buffers[SOFTWARE_DATA_BUFFER_COUNT + 2]
#define light_selector_data data_buffers[SOFTWARE_DATA_BUFFER_COUNT + 3]
#define light_metadatas data_buffers[SOFTWARE_DATA_BUFFER_COUNT + 4]
#define software_instances data_buffers[SOFTWARE_DATA_BUFFER_COUNT + 5]
#else
RaytracingAccelerationStructure as : register(t0, space2);
ByteAddressBuffer sobol_table : register(t0, space4);
ByteAddressBuffer camera_data : register(t0, space5);
ByteAddressBuffer data_buffers[] : register(t0, space6);
ByteAddressBuffer instance_metadatas : register(t0, space7);
ByteAddressBuffer light_selector_data : register(t0, space8);
ByteAddressBuffer light_metadatas : register(t0, space9);
Texture2D<float4> sdr_textures[] : register(t0, space10);
Texture2D<float4> hdr_textures[] : register(t0, space11);
SamplerState samplers[] : register(s0, space12);
#endif

float4 SampleTexture(int texture_index, float2 uv) {
  if (texture_index & 0x1000000) {
    return hdr_textures[NonUniformResourceIndex(texture_index & 0xFFFFFF)].SampleLevel(samplers[0], float2(uv.x, 1.0 - uv.y), 0.0);
  } else {
    return sdr_textures[NonUniformResourceIndex(texture_index)].SampleLevel(samplers[0], float2(uv.x, 1.0 - uv.y), 0.0);
  }
  return float4(1.0, 0.0, 1.0, 1.0);
}
