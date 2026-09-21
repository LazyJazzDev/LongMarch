#include "sparkium/backend/common/execution_backend.h"
#include "sparkium/backend/common/path_tracing/raytracing.h"
#include "sparkium/backend/cpu/scene_objects.h"

namespace sparkium::backend::cpu {
class RenderBackend final : public ExecutionBackend {
 public:
  using ExecutionBackend::ExecutionBackend;

  bool SupportsPipeline(RenderPipeline pipeline) const override {
    Info();  // Enforce the backend thread contract.
    return pipeline == RENDER_PIPELINE_AUTO || pipeline == RENDER_PIPELINE_RT_FALLBACK;
  }

 protected:
  RenderPipeline DefaultPipeline() const override {
    return RENDER_PIPELINE_RT_FALLBACK;
  }

  std::unique_ptr<backend::SceneObjects> CreateSceneObjects(Core *core,
                                                            std::shared_ptr<const SceneDefinition> scene) override {
    return std::make_unique<SceneObjects>(core, std::move(scene));
  }

  void Dispatch(Core *core, backend::SceneObjects *scene, RenderPipeline pipeline) override {
    raytracing::Render(core, scene->scene.get(), scene->camera.get(), scene->film.get(), true, false, false);
  }
};
}  // namespace sparkium::backend::cpu

namespace sparkium::backend {
std::unique_ptr<Backend> CreateCpuBackend(const RendererSettings &settings) {
  return std::make_unique<cpu::RenderBackend>(settings);
}
}  // namespace sparkium::backend
