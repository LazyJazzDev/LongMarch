// Umbrella for the HLSL-as-C++ compatibility layer used by the CPU backend.
//
// The shader sources are included inside `sparkium_cpu_shaders` so that the
// helpers they define (principled_util.hlsli declares its own `copysignf`, for
// instance) cannot collide with names from the C library or from grassland.
#pragma once

// The shaders are compiled once, for every scene, so their configuration is
// fixed here rather than supplied per scene the way the DXC invocations do it.
//
// SPARKIUM_SOFTWARE_RT selects the compute traversal the GPU fallback uses.
// The CPU backend replays those same traversal routines over host memory, so it
// wants the identical path. SPARKIUM_CPU_SHADER switches the few places that
// must diverge, such as resource declarations and compute entry points.
#ifndef SPARKIUM_SOFTWARE_RT
#define SPARKIUM_SOFTWARE_RT
#endif
#ifndef SPARKIUM_CPU_SHADER
#define SPARKIUM_CPU_SHADER
#endif

#include "sparkium/pipelines/raytracing/cpu/shaders/hlsl_cpu_resource.h"

namespace sparkium_cpu_shaders {
using namespace sparkium::cpu::hlsl;
}  // namespace sparkium_cpu_shaders
