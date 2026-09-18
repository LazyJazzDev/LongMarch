#pragma once

// The native CPU and CUDA rendering backends.
//
// Both run `pipelines/native/shared/*` -- a direct port of the HLSL shading
// core -- over a host-flattened copy of the scene. The CPU backend executes it
// on a thread pool; the CUDA backend executes the same functions inside a
// kernel. Neither uses the graphics device for rendering computation: the
// graphics core is only used to read back geometry, textures and to present the
// developed image.

#include "sparkium/core/core_util.h"

namespace sparkium::native {

enum Backend {
  BACKEND_CPU = 0,
  BACKEND_CUDA = 1,
};

// True when the build includes CUDA kernels and a usable device is present.
bool CudaAvailable();

// Renders one dispatch of `scene->settings.samples_per_dispatch` samples into
// the film, matching `raytracing::Render`.
void Render(sparkium::Core *core,
            sparkium::Scene *scene,
            sparkium::Camera *camera,
            sparkium::Film *film,
            Backend backend);

// Tone maps the native film into an 8-bit RGBA image without touching the
// graphics device, so a CPU-only run needs no GPU at any stage.
void DevelopToHost(sparkium::Film *film, std::vector<uint8_t> &pixels);

}  // namespace sparkium::native
