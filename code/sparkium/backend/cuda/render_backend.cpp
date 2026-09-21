#include "sparkium/backend/common/execution_backend.h"
#include "sparkium/backend/common/path_tracing/raytracing.h"
#include "sparkium/backend/cuda/scene_objects.h"

namespace sparkium::backend::cuda {
class RenderBackend final : public ExecutionBackend {
 public:
  using ExecutionBackend::ExecutionBackend;

  bool SupportsPipeline(RenderPipeline pipeline) const override {
    const auto info = Info();
    return pipeline == RENDER_PIPELINE_AUTO || pipeline == RENDER_PIPELINE_RT_FALLBACK ||
           (pipeline == RENDER_PIPELINE_RAY_TRACING && info.ray_tracing);
  }

 protected:
  RenderPipeline DefaultPipeline() const override {
    return Info().ray_tracing ? RENDER_PIPELINE_RAY_TRACING : RENDER_PIPELINE_RT_FALLBACK;
  }

  std::unique_ptr<backend::SceneObjects> CreateSceneObjects(Core *core,
                                                            std::shared_ptr<const SceneDefinition> scene) override {
    return std::make_unique<SceneObjects>(core, std::move(scene));
  }

  void Dispatch(Core *core, backend::SceneObjects *scene, RenderPipeline pipeline) override {
    raytracing::Render(core, scene->scene.get(), scene->camera.get(), scene->film.get(), true, false,
                       pipeline == RENDER_PIPELINE_RAY_TRACING);
  }
};
}  // namespace sparkium::backend::cuda

namespace sparkium::backend {
std::unique_ptr<Backend> CreateCudaBackend(const RendererSettings &settings) {
  return std::make_unique<cuda::RenderBackend>(settings);
}
}  // namespace sparkium::backend
