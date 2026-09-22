#pragma once
#include "common.hlsli"
#include "buffer_helper.hlsli"

RWTexture2D<float4> realtime_outputs[] : register(u0, space0);
#define accumulated_color realtime_outputs[0]
Texture2D<float4> history_inputs[] : register(t0, space1);
ConstantBuffer<RenderSettings> render_settings : register(b0, space3);
#define SOBOL_TABLE
ByteAddressBuffer software_nodes : register(t0, space2);
ByteAddressBuffer data_buffers[] : register(t0, space4);
#define sobol_table data_buffers[SOFTWARE_DATA_BUFFER_COUNT]
#define camera_data data_buffers[SOFTWARE_DATA_BUFFER_COUNT + 1]
#define instance_metadatas data_buffers[SOFTWARE_DATA_BUFFER_COUNT + 2]
#define light_selector_data data_buffers[SOFTWARE_DATA_BUFFER_COUNT + 3]
#define light_metadatas data_buffers[SOFTWARE_DATA_BUFFER_COUNT + 4]
#define software_instances data_buffers[SOFTWARE_DATA_BUFFER_COUNT + 5]
Texture2D<float4> sdr_textures[] : register(t0, space5);
Texture2D<float4> hdr_textures[] : register(t0, space6);
SamplerState samplers[] : register(s0, space7);

float4 SampleTexture(int texture_index, float2 uv) {
  if (texture_index & 0x1000000) {
    return hdr_textures[NonUniformResourceIndex(texture_index & 0xFFFFFF)].SampleLevel(samplers[0],
                                                                                       float2(uv.x, 1.0 - uv.y), 0.0);
  } else {
    return sdr_textures[NonUniformResourceIndex(texture_index)].SampleLevel(samplers[0], float2(uv.x, 1.0 - uv.y), 0.0);
  }
  return float4(1.0, 0.0, 1.0, 1.0);
}
