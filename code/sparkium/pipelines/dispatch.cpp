#include "sparkium/core/core.h"
#include "sparkium/core/film.h"
#include "sparkium/pipelines/pipelines.h"

namespace sparkium {
RenderPipeline Core::ResolveRenderPipeline(RenderPipeline render_pipeline) const {
  if (render_pipeline == RENDER_PIPELINE_AUTO) {
    const bool prefer_ray_query =
        core_->API() == graphics::BACKEND_API_D3D12 || core_->API() == graphics::BACKEND_API_VULKAN;
    if (prefer_ray_query && core_->DeviceRayQuerySupport()) {
      render_pipeline = RENDER_PIPELINE_RAY_QUERY;
    } else if (core_->DeviceRayTracingSupport()) {
      render_pipeline = RENDER_PIPELINE_RAY_TRACING;
    } else if (core_->DeviceRayQuerySupport()) {
      render_pipeline = RENDER_PIPELINE_RAY_QUERY;
    } else {
      render_pipeline = RENDER_PIPELINE_RT_FALLBACK;
    }
  }
  // Older Blender scenes request pipeline RT. Keep them on native traversal
  // when the device supports inline queries instead of a full RT pipeline.
  if (render_pipeline == RENDER_PIPELINE_RAY_TRACING && !core_->DeviceRayTracingSupport())
    return core_->DeviceRayQuerySupport() ? RENDER_PIPELINE_RAY_QUERY : RENDER_PIPELINE_RT_FALLBACK;
  return render_pipeline;
}

void Core::Render(Scene *scene, Camera *camera, Film *film, RenderPipeline render_pipeline) {
  render_pipeline = ResolveRenderPipeline(render_pipeline);
  if (film->last_pipeline_ != render_pipeline) {
    film->Reset();
    film->last_pipeline_ = render_pipeline;
  }
  switch (render_pipeline) {
    case RENDER_PIPELINE_REALTIME:
      realtime::Render(this, scene, camera, film);
      break;
    case RENDER_PIPELINE_RAY_TRACING:
      raytracing::Render(this, scene, camera, film);
      break;
    case RENDER_PIPELINE_RAY_QUERY:
      if (!core_->DeviceRayQuerySupport())
        throw std::runtime_error("ray_query is unavailable on the selected graphics backend");
      raytracing::Render(this, scene, camera, film, true, true);
      break;
    case RENDER_PIPELINE_RT_FALLBACK:
      raytracing::Render(this, scene, camera, film, true);
      break;
    default:
      LogError("Unknown render pipeline");
  }
}

}  // namespace sparkium
