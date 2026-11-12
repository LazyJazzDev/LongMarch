#pragma once
#include "common.slang"
#include "buffer_helper.slang"
#include "render_binding_slots.h"

RHI_RW_TEXTURE2D(SPARKIUM_GROUP_FILM, SPARKIUM_FILM_COLOR, float4, accumulated_color);
RHI_RW_TEXTURE2D(SPARKIUM_GROUP_FILM, SPARKIUM_FILM_SAMPLES, float, accumulated_samples);
RHI_UNIFORM(SPARKIUM_GROUP_RENDERER, SPARKIUM_RENDERER_SETTINGS, RenderSettings, render_settings);
RHI_BYTE_ADDRESS_BUFFER(SPARKIUM_GROUP_RENDERER, SPARKIUM_RENDERER_SOBOL, sobol_table);
#define SOBOL_TABLE
RHI_BYTE_ADDRESS_BUFFER(SPARKIUM_GROUP_CAMERA, SPARKIUM_CAMERA_DATA, camera_data);

#ifdef SPARKIUM_SOFTWARE_RT
#ifdef SPARKIUM_RAY_QUERY
RHI_ACCELERATION_STRUCTURE(SPARKIUM_GROUP_SCENE, SPARKIUM_SCENE_TRAVERSAL, query_scene);
#else
RHI_BYTE_ADDRESS_BUFFER(SPARKIUM_GROUP_SCENE, SPARKIUM_SCENE_TRAVERSAL, software_nodes);
#endif
#else
RHI_ACCELERATION_STRUCTURE(SPARKIUM_GROUP_SCENE, SPARKIUM_SCENE_TRAVERSAL, as);
#endif
RHI_BYTE_ADDRESS_BUFFER_ARRAY(SPARKIUM_GROUP_SCENE, SPARKIUM_SCENE_BUFFERS, data_buffers);
RHI_BYTE_ADDRESS_BUFFER(SPARKIUM_GROUP_SCENE, SPARKIUM_SCENE_INSTANCES, instance_metadatas);
RHI_BYTE_ADDRESS_BUFFER(SPARKIUM_GROUP_SCENE, SPARKIUM_SCENE_LIGHT_SELECTOR, light_selector_data);
RHI_BYTE_ADDRESS_BUFFER(SPARKIUM_GROUP_SCENE, SPARKIUM_SCENE_LIGHT_METADATA, light_metadatas);
RHI_BYTE_ADDRESS_BUFFER(SPARKIUM_GROUP_SCENE, SPARKIUM_SCENE_SOFTWARE_INSTANCES, software_instances);
RHI_TEXTURE2D_ARRAY(SPARKIUM_GROUP_SCENE, SPARKIUM_SCENE_SDR_TEXTURES, float4, sdr_textures);
RHI_TEXTURE2D_ARRAY(SPARKIUM_GROUP_SCENE, SPARKIUM_SCENE_HDR_TEXTURES, float4, hdr_textures);
RHI_SAMPLER_ARRAY(SPARKIUM_GROUP_SCENE, SPARKIUM_SCENE_SAMPLERS, samplers);

float4 SampleTexture(int texture_index, float2 uv) {
  if ((texture_index & 0x1000000) != 0) {
    return hdr_textures[NonUniformResourceIndex(texture_index & 0xFFFFFF)].SampleLevel(samplers[0],
                                                                                       float2(uv.x, 1.0 - uv.y), 0.0);
  } else {
    return sdr_textures[NonUniformResourceIndex(texture_index)].SampleLevel(samplers[0], float2(uv.x, 1.0 - uv.y), 0.0);
  }
  return float4(1.0, 0.0, 1.0, 1.0);
}
