#pragma once

#include <string>
#include <vector>

#include "sparkium/pipelines/portable/scene_bake.h"

namespace sparkium::portable {

// Returns the transpiled portable render kernel for the baked materials. The
// same source is compiled for the host and for CUDA. It contains
//   void PortableRenderPixel(uint32_t x, uint32_t y)
// plus all shading code, and expects hlsl_compat.h to be included first.
std::string GenerateKernelSource(sparkium::Core *core, const std::vector<MaterialData> &materials);

// Compiles (at build time or first use) the kernel sources into the CPU and
// CUDA entry points. Implemented in kernel_cpu.cpp / kernel_cuda.cu through
// generated includes.
void CompileKernelCache(sparkium::Core *core, const BakeResult &bake);

}  // namespace sparkium::portable
