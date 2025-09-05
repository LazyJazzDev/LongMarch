#pragma once
#include "common.hlsli"
#include "buffer_helper.hlsli"

RWTexture2D<float4> accumulated_color : register(u0, space0);
RWTexture2D<float> accumulated_samples : register(u0, space1);

RaytracingAccelerationStructure as : register(t0, space2);

ConstantBuffer<RenderSettings> render_settings : register(b0, space3);

#define SOBOL_TABLE
StructuredBuffer<uint> sobol_table : register(t0, space4);

ByteAddressBuffer camera_data : register(t0, space5);
ByteAddressBuffer data_buffers[] : register(t0, space6);
StructuredBuffer<InstanceMetadata> instance_metadatas : register(t0, space7);

ByteAddressBuffer light_selector_data : register(t0, space8);
StructuredBuffer<LightMetadata> light_metadatas : register(t0, space9);

Texture2D<float4> sdr_textures[] : register(t0, space10);
Texture2D<float4> hdr_textures[] : register(t0, space11);
SamplerState samplers[] : register(s0, space12);

float4 SampleTexture(int texture_index, float2 uv) {
  if (texture_index & 0x1000000) {
    return hdr_textures[texture_index & 0xFFFFFF].SampleLevel(samplers[0], float2(uv.x, 1.0 - uv.y), 0.0);
  } else {
    return sdr_textures[texture_index].SampleLevel(samplers[0], float2(uv.x, 1.0 - uv.y), 0.0);
  }
  return float4(1.0, 0.0, 1.0, 1.0);
}
