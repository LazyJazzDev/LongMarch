#pragma once

#include "sparkium/pipelines/portable/scene_bake.h"

namespace sparkium::portable {

// Owning host-side storage behind a KernelContext. The context's data buffer
// table is an array of N byte pointers immediately followed by N sizes (see
// hlsl_compat.h ResourceDataBuffer).
struct FrameResources {
  sparkium_portable::KernelContext ctx{};
  std::vector<const uint8_t *> table;  // N pointers, then N sizes in place
  std::vector<uint8_t> instance_metadatas;
  std::vector<uint8_t> light_metadatas;
  std::vector<uint8_t> light_selector;
  std::vector<uint8_t> instances;
  std::vector<sparkium_portable::Texture2D> sdr_textures;
  std::vector<sparkium_portable::Texture2D> hdr_textures;
  std::vector<uint32_t> sobol_bits;
};

void PopulateContext(const BakeResult &bake,
                     const std::vector<uint32_t> &sobol_table,
                     sparkium_portable::float4 *color,
                     float *samples,
                     uint32_t width,
                     uint32_t height,
                     FrameResources &res);

}  // namespace sparkium::portable
