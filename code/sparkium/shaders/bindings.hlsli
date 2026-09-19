#include "native_contract.hlsli"
#pragma once
#include "common.hlsli"
#include "buffer_helper.hlsli"

SP_RESOURCE(SP_RW_TEXTURE(float4), accumulated_color, u0, 0);
#define SP_BINDING_accumulated_color SP_RESOURCE_ACCESS(SP_RW_TEXTURE(float4), accumulated_color, 0)
SP_RESOURCE(SP_RW_TEXTURE(float), accumulated_samples, u0, 1);
#define SP_BINDING_accumulated_samples SP_RESOURCE_ACCESS(SP_RW_TEXTURE(float), accumulated_samples, 1)
SP_RESOURCE(ConstantBuffer<RenderSettings>, render_settings, b0, 3);
#define SP_BINDING_render_settings SP_RESOURCE_ACCESS(ConstantBuffer<RenderSettings>, render_settings, 3)
#define SOBOL_TABLE
#ifdef SPARKIUM_SOFTWARE_RT
#ifdef SPARKIUM_RAY_QUERY
RaytracingAccelerationStructure query_scene : register(t0, space2);
#else
SP_RESOURCE(ByteAddressBuffer, software_nodes, t0, 2);
#define SP_BINDING_software_nodes SP_RESOURCE_ACCESS(ByteAddressBuffer, software_nodes, 2)
#endif
SP_ARRAY_RESOURCE(ByteAddressBuffer, data_buffers, t0, 4);
#define SP_BINDING_data_buffers SP_ARRAY_ACCESS(ByteAddressBuffer, data_buffers, 4)
#define SP_BINDING_sobol_table SP_BINDING_data_buffers[SOFTWARE_DATA_BUFFER_COUNT]
#define SP_BINDING_camera_data SP_BINDING_data_buffers[SOFTWARE_DATA_BUFFER_COUNT + 1]
#define SP_BINDING_instance_metadatas SP_BINDING_data_buffers[SOFTWARE_DATA_BUFFER_COUNT + 2]
#define SP_BINDING_light_selector_data SP_BINDING_data_buffers[SOFTWARE_DATA_BUFFER_COUNT + 3]
#define SP_BINDING_light_metadatas SP_BINDING_data_buffers[SOFTWARE_DATA_BUFFER_COUNT + 4]
#define SP_BINDING_software_instances SP_BINDING_data_buffers[SOFTWARE_DATA_BUFFER_COUNT + 5]
SP_ARRAY_RESOURCE(SP_TEXTURE(float4), sdr_textures, t0, 5);
#define SP_BINDING_sdr_textures SP_ARRAY_ACCESS(SP_TEXTURE(float4), sdr_textures, 5)
SP_ARRAY_RESOURCE(SP_TEXTURE(float4), hdr_textures, t0, 6);
#define SP_BINDING_hdr_textures SP_ARRAY_ACCESS(SP_TEXTURE(float4), hdr_textures, 6)
SP_ARRAY_RESOURCE(SP_SAMPLER, samplers, s0, 7);
#define SP_BINDING_samplers SP_ARRAY_ACCESS(SP_SAMPLER, samplers, 7)
#else
RaytracingAccelerationStructure as : register(t0, space2);
SP_RESOURCE(ByteAddressBuffer, sobol_table, t0, 4);
#define SP_BINDING_sobol_table SP_RESOURCE_ACCESS(ByteAddressBuffer, sobol_table, 4)
SP_RESOURCE(ByteAddressBuffer, camera_data, t0, 5);
#define SP_BINDING_camera_data SP_RESOURCE_ACCESS(ByteAddressBuffer, camera_data, 5)
SP_ARRAY_RESOURCE(ByteAddressBuffer, data_buffers, t0, 6);
#define SP_BINDING_data_buffers SP_ARRAY_ACCESS(ByteAddressBuffer, data_buffers, 6)
SP_RESOURCE(ByteAddressBuffer, instance_metadatas, t0, 7);
#define SP_BINDING_instance_metadatas SP_RESOURCE_ACCESS(ByteAddressBuffer, instance_metadatas, 7)
SP_RESOURCE(ByteAddressBuffer, light_selector_data, t0, 8);
#define SP_BINDING_light_selector_data SP_RESOURCE_ACCESS(ByteAddressBuffer, light_selector_data, 8)
SP_RESOURCE(ByteAddressBuffer, light_metadatas, t0, 9);
#define SP_BINDING_light_metadatas SP_RESOURCE_ACCESS(ByteAddressBuffer, light_metadatas, 9)
SP_ARRAY_RESOURCE(SP_TEXTURE(float4), sdr_textures, t0, 10);
#define SP_BINDING_sdr_textures SP_ARRAY_ACCESS(SP_TEXTURE(float4), sdr_textures, 10)
SP_ARRAY_RESOURCE(SP_TEXTURE(float4), hdr_textures, t0, 11);
#define SP_BINDING_hdr_textures SP_ARRAY_ACCESS(SP_TEXTURE(float4), hdr_textures, 11)
SP_ARRAY_RESOURCE(SP_SAMPLER, samplers, s0, 12);
#define SP_BINDING_samplers SP_ARRAY_ACCESS(SP_SAMPLER, samplers, 12)
#endif

float4 SampleTexture(SP_CONTEXT int texture_index, float2 uv) {
  if (texture_index & 0x1000000) {
    return SP_BINDING_hdr_textures[SP_NONUNIFORM(texture_index & 0xFFFFFF)].SampleLevel(SP_BINDING_samplers[0],
                                                                                        float2(uv.x, 1.0 - uv.y), 0.0);
  } else {
    return SP_BINDING_sdr_textures[SP_NONUNIFORM(texture_index)].SampleLevel(SP_BINDING_samplers[0],
                                                                             float2(uv.x, 1.0 - uv.y), 0.0);
  }
  return float4(1.0, 0.0, 1.0, 1.0);
}
