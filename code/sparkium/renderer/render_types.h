#pragma once

namespace sparkium {
enum class RenderBackend { Graphics, CPU, CUDA };
enum class GraphicsAPI { Default, D3D12, Vulkan, Metal };

typedef enum RenderPipeline {
  RENDER_PIPELINE_RASTERIZATION = 0,
  RENDER_PIPELINE_RAY_TRACING = 1,
  RENDER_PIPELINE_AUTO = 2,
  RENDER_PIPELINE_RT_FALLBACK = 3,  // Compute BVH traversal without hardware ray tracing
  RENDER_PIPELINE_RAY_QUERY = 4     // Compute path tracing with hardware acceleration structures
} RenderPipeline;
}  // namespace sparkium
