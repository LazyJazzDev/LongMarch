#pragma once

// Shared CPU/CUDA rendering backend for Sparkium. The shading kernels are
// transpiled once from the HLSL sources in code/sparkium/shaders and compiled
// for the host (always) and for CUDA (when enabled); both execute identical
// generated code.

#include "sparkium/core/core_util.h"
#include "sparkium/core/camera.h"
#include "sparkium/core/film.h"
#include "sparkium/core/scene.h"
#include "sparkium/scene_io/json_scene.h"

namespace sparkium::portable {

// True when the library was built with CUDA support and a CUDA device is
// usable. Kept as a runtime query so a CUDA-less binary still loads.
bool CudaAvailable();

// Number of worker threads used by the host backend (defaults to hardware
// concurrency, overridable with SPARKIUM_CPU_THREADS).
uint32_t HostThreadCount();

// Renders one frame of `scene` into the film accumulation buffers of
// `accumulation` with `kind`. The accumulation vectors hold float4 color and
// float sample counts, width*height each, and are preserved across frames so
// progressive accumulation matches the GPU pipelines.
void RenderFrame(ComputeBackendKind kind,
                 sparkium::Scene *scene,
                 sparkium::Camera *camera,
                 sparkium::Film *film,
                 const std::map<graphics::Image *, const HostImageData *> &host_images,
                 const std::vector<uint32_t> &sobol_table,
                 uint32_t *seed,
                 std::vector<float> &accumulation_color,
                 std::vector<float> &accumulation_samples);

// Applies the film tone mapping (shaders/tone_mapping.hlsl semantics) and
// writes 8-bit RGBA.
void Develop(const sparkium::Film *film,
             const std::vector<float> &accumulation_color,
             const std::vector<float> &accumulation_samples,
             std::vector<uint8_t> &rgba);

}  // namespace sparkium::portable
