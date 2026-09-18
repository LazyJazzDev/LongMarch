// Entry point of the CPU path tracer.
//
// software/render.hlsl is the same shader the GPU fallback compiles; including
// it here compiles its traversal loop, materials and BSDFs into native code.
// Everything it needs is already in scope from hlsl_cpu_materials.h.
#pragma once

#include "sparkium/pipelines/raytracing/cpu/shaders/hlsl_cpu_materials.h"

namespace sparkium_cpu_shaders {

#include "software/render.hlsl"

}  // namespace sparkium_cpu_shaders
