#pragma once
#include <webgpu/webgpu_cpp.h>

#include "grassland/graphics/interface.h"

namespace grassland::graphics::backend {

wgpu::TextureFormat WebGPUFormat(ImageFormat format);

// Bytes per texel as stored on the GPU. RGB32F is widened to RGBA32F, which
// WebGPU does support.
uint32_t WebGPUTexelSize(ImageFormat format);

class WebGPUCore;
class WebGPUBuffer;
class WebGPUImage;
class WebGPUSampler;
class WebGPUShader;
class WebGPUProgram;
class WebGPUComputeProgram;
class WebGPUCommandContext;

struct WebGPUBinding {
  ResourceType type;
  int count;
};

}  // namespace grassland::graphics::backend
