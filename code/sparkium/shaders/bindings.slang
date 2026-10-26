#pragma once
#include "common.slang"
#include "buffer_helper.slang"

RHI_RW_TEXTURE2D(0, 0, float4, accumulated_color);
RHI_RW_TEXTURE2D(0, 1, float, accumulated_samples);
RHI_UNIFORM(0, 3, RenderSettings, render_settings);
#define SOBOL_TABLE
#ifdef SPARKIUM_SOFTWARE_RT
#ifdef SPARKIUM_RAY_QUERY
RHI_ACCELERATION_STRUCTURE(0, 2, query_scene);
#else
RHI_BYTE_ADDRESS_BUFFER(0, 2, software_nodes);
#endif
RHI_BYTE_ADDRESS_BUFFER_ARRAY(0, 4, data_buffers);
#define sobol_table data_buffers[SOFTWARE_DATA_BUFFER_COUNT]
#define camera_data data_buffers[SOFTWARE_DATA_BUFFER_COUNT + 1]
#define instance_metadatas data_buffers[SOFTWARE_DATA_BUFFER_COUNT + 2]
#define light_selector_data data_buffers[SOFTWARE_DATA_BUFFER_COUNT + 3]
#define light_metadatas data_buffers[SOFTWARE_DATA_BUFFER_COUNT + 4]
#define software_instances data_buffers[SOFTWARE_DATA_BUFFER_COUNT + 5]
RHI_TEXTURE2D_ARRAY(0, 5, float4, sdr_textures);
RHI_TEXTURE2D_ARRAY(0, 6, float4, hdr_textures);
RHI_SAMPLER_ARRAY(0, 7, samplers);
#else
RHI_ACCELERATION_STRUCTURE(0, 2, as);
RHI_BYTE_ADDRESS_BUFFER(0, 4, sobol_table);
RHI_BYTE_ADDRESS_BUFFER(0, 5, camera_data);
RHI_BYTE_ADDRESS_BUFFER_ARRAY(0, 6, data_buffers);
RHI_BYTE_ADDRESS_BUFFER(0, 7, instance_metadatas);
RHI_BYTE_ADDRESS_BUFFER(0, 8, light_selector_data);
RHI_BYTE_ADDRESS_BUFFER(0, 9, light_metadatas);
RHI_TEXTURE2D_ARRAY(0, 10, float4, sdr_textures);
RHI_TEXTURE2D_ARRAY(0, 11, float4, hdr_textures);
RHI_SAMPLER_ARRAY(0, 12, samplers);
#endif

float4 SampleTexture(int texture_index, float2 uv) {
  if ((texture_index & 0x1000000) != 0) {
    return hdr_textures[NonUniformResourceIndex(texture_index & 0xFFFFFF)].SampleLevel(samplers[0],
                                                                                       float2(uv.x, 1.0 - uv.y), 0.0);
  } else {
    return sdr_textures[NonUniformResourceIndex(texture_index)].SampleLevel(samplers[0], float2(uv.x, 1.0 - uv.y), 0.0);
  }
  return float4(1.0, 0.0, 1.0, 1.0);
}
