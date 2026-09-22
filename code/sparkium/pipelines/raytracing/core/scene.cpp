#include "sparkium/pipelines/raytracing/core/scene.h"

#include "grassland/graphics/frame_profile.h"
#include "sparkium/pipelines/common/core/camera.h"
#include "sparkium/pipelines/common/core/core.h"
#include "sparkium/pipelines/raytracing/core/film.h"

namespace sparkium::raytracing {
void Scene::Render(render_shared::Camera *camera, Film *film, bool software, bool ray_query) {
  software = software || ray_query;
  if (ray_query != ray_query_) {
    ray_query_ = ray_query;
    software_pipeline_.reset();
    pipeline_dirty_ = true;
    if (rendered_)
      film->Reset();
  }
  if (software != software_tracing_) {
    software_tracing_ = software;
    pipeline_dirty_ = true;
    if (rendered_)
      film->Reset();
  }
  if (software && !software_pipeline_)
    software_pipeline_ = std::make_unique<render_shared::SoftwarePipeline>(core_, ray_query_);
  graphics::CpuProfileScope update_profile("scene_update");
  UpdatePipeline(camera);
  update_profile.End();
  graphics::CpuProfileScope setup_profile("render_setup");
  rendered_ = true;
  scene_settings_buffer_->UploadData(&settings.raytracing, sizeof(Settings::RayTracing));
  scene_settings_buffer_->UploadData(&film->film_.info, sizeof(sparkium::Film::Info), sizeof(Settings::RayTracing));
  film->film_.info.accumulated_samples += settings.raytracing.samples_per_dispatch;
  std::unique_ptr<graphics::CommandContext> cmd_context;
  core_->GraphicsCore()->CreateCommandContext(&cmd_context);
  graphics::GpuProfileScope trace_profile(cmd_context.get(), "path_trace");
  const auto bind_point = software ? graphics::BIND_POINT_COMPUTE : graphics::BIND_POINT_RAYTRACING;
  if (software)
    cmd_context->CmdBindComputeProgram(software_pipeline_->Program());
  else
    cmd_context->CmdBindRayTracingProgram(rt_program_.get());
  cmd_context->CmdBindResources(0, {film->accumulated_color_.get()}, bind_point);
  cmd_context->CmdBindResources(1, {film->accumulated_samples_.get()}, bind_point);
  if (software) {
    if (ray_query_)
      cmd_context->CmdBindResources(2, software_pipeline_->AccelerationStructure(), bind_point);
    else
      cmd_context->CmdBindResources(2, {software_pipeline_->Nodes()}, bind_point);
  } else
    cmd_context->CmdBindResources(2, tlas_.get(), bind_point);
  cmd_context->CmdBindResources(3, {scene_settings_buffer_.get()}, bind_point);
  if (software) {
    auto resources = buffers_;
    resources.insert(resources.end(),
                     {core_->GetBuffer("sobol"), camera->Buffer(), instance_metadata_buffer_.get(),
                      light_selector_buffer_.get(), light_metadatas_buffer_.get(), software_pipeline_->Instances()});
    cmd_context->CmdBindResources(4, resources, bind_point);
    cmd_context->CmdBindResources(5, sdr_images_, bind_point);
    cmd_context->CmdBindResources(6, hdr_images_, bind_point);
    cmd_context->CmdBindResources(7, std::vector{linear_sampler_.get(), nearest_sampler_.get()}, bind_point);
  } else {
    cmd_context->CmdBindResources(4, {core_->GetBuffer("sobol")}, bind_point);
    cmd_context->CmdBindResources(5, {camera->Buffer()}, bind_point);
    cmd_context->CmdBindResources(6, buffers_, bind_point);
    cmd_context->CmdBindResources(7, {instance_metadata_buffer_.get()}, bind_point);
    cmd_context->CmdBindResources(8, {light_selector_buffer_.get()}, bind_point);
    cmd_context->CmdBindResources(9, {light_metadatas_buffer_.get()}, bind_point);
    cmd_context->CmdBindResources(10, sdr_images_, bind_point);
    cmd_context->CmdBindResources(11, hdr_images_, bind_point);
    cmd_context->CmdBindResources(12, std::vector{linear_sampler_.get(), nearest_sampler_.get()}, bind_point);
  }
  if (software)
    cmd_context->CmdDispatch((film->GetWidth() + 7) / 8, (film->GetHeight() + 7) / 8, 1);
  else
    cmd_context->CmdDispatchRays(film->accumulated_color_->Extent().width, film->accumulated_samples_->Extent().height,
                                 1);

  trace_profile.End();
  graphics::GpuProfileScope resolve_profile(cmd_context.get(), "film_resolve");
  cmd_context->CmdBindComputeProgram(core_->GetComputeProgram("film2img"));
  cmd_context->CmdBindResources(0, {film->accumulated_color_.get()}, graphics::BIND_POINT_COMPUTE);
  cmd_context->CmdBindResources(1, {film->accumulated_samples_.get()}, graphics::BIND_POINT_COMPUTE);
  cmd_context->CmdBindResources(2, {film->film_.GetRawImage()}, graphics::BIND_POINT_COMPUTE);
  cmd_context->CmdDispatch((film->film_.GetRawImage()->Extent().width + 7) / 8,
                           (film->film_.GetRawImage()->Extent().height + 7) / 8, 1);

  resolve_profile.End();
  setup_profile.End();
  graphics::CpuProfileScope submit_profile("render_submit");
  core_->GraphicsCore()->SubmitCommandContext(cmd_context.get());
  submit_profile.End();
  graphics::CpuProfileScope wait_profile("render_wait");
  core_->GraphicsCore()->WaitGPU();
}

Scene *DedicatedCast(sparkium::Scene *scene) {
  COMPONENT_CAST(scene, Scene);
}
}  // namespace sparkium::raytracing
