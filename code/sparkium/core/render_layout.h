#pragma once

#include "grassland/rhi/rhi_binding_group.h"
#include "sparkium/shaders/render_binding_slots.h"

namespace sparkium {

// Group numbers describe ownership, not individual resource kinds. Array
// capacities belong to a program version; growing them replaces its layout.
inline grassland::rhi::PipelineLayoutDesc RayTracingLayout(bool software_bvh,
                                                           uint32_t buffer_count,
                                                           uint32_t sdr_count,
                                                           uint32_t hdr_count) {
  using namespace grassland::rhi;
  const auto capacity = [](uint32_t count) { return count ? count : 1u; };
  PipelineLayoutDesc layout;
  layout.groups.resize(SPARKIUM_GROUP_COUNT);
  layout.groups[SPARKIUM_GROUP_FILM].bindings = {{SPARKIUM_FILM_COLOR, BINDING_RW_TEXTURE2D},
                                                 {SPARKIUM_FILM_SAMPLES, BINDING_RW_TEXTURE2D}};
  layout.groups[SPARKIUM_GROUP_SCENE].bindings = {
      {SPARKIUM_SCENE_TRAVERSAL, software_bvh ? BINDING_BYTE_ADDRESS_BUFFER : BINDING_ACCELERATION_STRUCTURE},
      {SPARKIUM_SCENE_BUFFERS, BINDING_BYTE_ADDRESS_BUFFER, capacity(buffer_count)},
      {SPARKIUM_SCENE_INSTANCES, BINDING_BYTE_ADDRESS_BUFFER},
      {SPARKIUM_SCENE_LIGHT_SELECTOR, BINDING_BYTE_ADDRESS_BUFFER},
      {SPARKIUM_SCENE_LIGHT_METADATA, BINDING_BYTE_ADDRESS_BUFFER},
      {SPARKIUM_SCENE_SOFTWARE_INSTANCES, BINDING_BYTE_ADDRESS_BUFFER},
      {SPARKIUM_SCENE_SDR_TEXTURES, BINDING_TEXTURE2D, capacity(sdr_count)},
      {SPARKIUM_SCENE_HDR_TEXTURES, BINDING_TEXTURE2D, capacity(hdr_count)},
      {SPARKIUM_SCENE_SAMPLERS, BINDING_SAMPLER, 2}};
  layout.groups[SPARKIUM_GROUP_CAMERA].bindings = {{SPARKIUM_CAMERA_DATA, BINDING_BYTE_ADDRESS_BUFFER}};
  layout.groups[SPARKIUM_GROUP_RENDERER].bindings = {{SPARKIUM_RENDERER_SETTINGS, BINDING_UNIFORM_BUFFER},
                                                     {SPARKIUM_RENDERER_SOBOL, BINDING_BYTE_ADDRESS_BUFFER}};
  return layout;
}

}  // namespace sparkium
