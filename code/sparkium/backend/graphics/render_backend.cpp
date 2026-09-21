#include "sparkium/backend/common/execution_backend.h"
#include "sparkium/backend/common/path_tracing/raytracing.h"
#include "sparkium/backend/graphics/raster/raster.h"
#include "sparkium/backend/graphics/scene_objects.h"

namespace sparkium::backend::graphics_backend {
class RenderBackend final : public ExecutionBackend {
 public:
  using ExecutionBackend::ExecutionBackend;

  bool SupportsPipeline(RenderPipeline pipeline) const override {
    const auto info = Info();
    return pipeline == RENDER_PIPELINE_AUTO || pipeline == RENDER_PIPELINE_RT_FALLBACK ||
           pipeline == RENDER_PIPELINE_RASTERIZATION || (pipeline == RENDER_PIPELINE_RAY_QUERY && info.ray_query) ||
           (pipeline == RENDER_PIPELINE_RAY_TRACING && info.ray_tracing);
  }

 protected:
  RenderPipeline DefaultPipeline() const override {
    const auto info = Info();
    auto api = BackendDevice()->GraphicsCore()->API();
    if ((api == graphics::BACKEND_API_D3D12 || api == graphics::BACKEND_API_VULKAN) && info.ray_query)
      return RENDER_PIPELINE_RAY_QUERY;
    if (info.ray_tracing)
      return RENDER_PIPELINE_RAY_TRACING;
    return info.ray_query ? RENDER_PIPELINE_RAY_QUERY : RENDER_PIPELINE_RT_FALLBACK;
  }

  std::unique_ptr<backend::SceneObjects> CreateSceneObjects(Core *core,
                                                            std::shared_ptr<const SceneDefinition> scene) override {
    return std::make_unique<SceneObjects>(core, std::move(scene));
  }

  void Dispatch(Core *core, backend::SceneObjects *scene, RenderPipeline pipeline) override {
    if (pipeline == RENDER_PIPELINE_RASTERIZATION)
      raster::Render(core, scene->scene.get(), scene->camera.get(), scene->film.get());
    else
      raytracing::Render(core, scene->scene.get(), scene->camera.get(), scene->film.get(),
                         pipeline != RENDER_PIPELINE_RAY_TRACING, pipeline == RENDER_PIPELINE_RAY_QUERY);
  }
};
}  // namespace sparkium::backend::graphics_backend

namespace sparkium::backend {
std::unique_ptr<Backend> CreateGraphicsBackend(const RendererSettings &settings) {
  return std::make_unique<graphics_backend::RenderBackend>(settings);
}
}  // namespace sparkium::backend
