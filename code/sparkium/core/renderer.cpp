#include "sparkium/core/renderer.h"

#include "sparkium/core/camera.h"
#include "sparkium/core/film.h"
#include "sparkium/core/render_layout.h"
#include "sparkium/core/scene.h"

namespace sparkium {

Renderer::Renderer(Core *core, rhi::QueueType queue)
    : core_(core),
      queue_(core ? core->GetDevice()->GetQueue(queue) : nullptr) {
}

bool Renderer::IsValid() const {
  return core_ && core_->IsValid() && queue_;
}

rhi::IBindingGroup *Renderer::RecordParameters(rhi::ICommandContext *commands, const Scene *scene, const Film *film) {
  if (!IsValid() || !commands || !scene || !film)
    return nullptr;
  if (!parameters_)
    parameters_ = core_->GetDevice()->CreateBuffer(
        {(sizeof(Scene::Settings::RayTracing) + sizeof(Film::Info) + 255) & ~uint64_t(255)});
  if (!bindings_)
    bindings_ =
        core_->GetDevice()->CreateBindingGroup(RayTracingLayout(false, 1, 1, 1).groups[SPARKIUM_GROUP_RENDERER]);
  if (!parameters_ || !bindings_ || !bindings_->BindResource(SPARKIUM_RENDERER_SETTINGS, parameters_.Get()) ||
      !bindings_->BindResource(SPARKIUM_RENDERER_SOBOL, core_->GetBuffer("sobol")) ||
      !commands->CmdUploadBuffer(parameters_.Get(), 0, &scene->settings.raytracing,
                                 sizeof(Scene::Settings::RayTracing)) ||
      !commands->CmdUploadBuffer(parameters_.Get(), sizeof(Scene::Settings::RayTracing), &film->info,
                                 sizeof(Film::Info)))
    return nullptr;
  return bindings_.Get();
}

rhi::ObjPtr<rhi::ISubmissionToken> Renderer::Submit(Scene *scene,
                                                    Camera *camera,
                                                    Film *film,
                                                    RenderPipeline pipeline,
                                                    rhi::IImage *output,
                                                    bool linear_hdr,
                                                    const rhi::SubmitInfo &dependencies) {
  if (!IsValid() || !scene || !camera || !film || !film->IsValid() || scene->GetCore() != core_ ||
      camera->GetCore() != core_ || film->GetCore() != core_)
    return {};
  const auto resolved = core_->ResolveRenderPipeline(pipeline);
  const int samples = resolved == RENDER_PIPELINE_RASTERIZATION ? 1 : scene->settings.samples_per_dispatch;
  if (samples <= 0)
    return {};
  rhi::ObjPtr<rhi::ICommandContext> commands = queue_->CreateCommandContext();
  auto updates = core_->BeginUpdates();
  if (!commands || !core_->RecordRender(*this, commands.Get(), updates, scene, camera, film, resolved) ||
      (output && !film->RecordDevelop(commands.Get(), output, linear_hdr))) {
    // Recording may have changed cached AS/pipeline state. Do not reuse that
    // state after a failed transaction; no CPU sample progress is accepted.
    core_->ReportFailure("Cannot record render submission");
    return {};
  }
  rhi::ObjPtr<rhi::ISubmissionToken> submission = commands->Submit(dependencies);
  if (!submission) {
    core_->ReportFailure("Cannot submit render commands");
    return {};
  }
  if (!core_->AcceptUpdates(updates)) {
    core_->ReportFailure("Pending resource updates changed during submission");
    return submission;
  }
  film->AcceptRender(samples);
  return submission;
}

}  // namespace sparkium
