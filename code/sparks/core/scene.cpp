#include "sparks/core/scene.h"

#include "core.h"
#include "film.h"

namespace sparks {
Scene::Scene(Core *core) : core_(core) {
}

void Scene::Render(Camera *camera, Film *film) {
  UpdatePipeline();
  std::unique_ptr<graphics::CommandContext> cmd_context;
  core_->GraphicsCore()->CreateCommandContext(&cmd_context);
  cmd_context->CmdBindRayTracingProgram(rt_program_.get());
  cmd_context->CmdBindResources(0, {film->accumulated_color_.get()}, graphics::BIND_POINT_RAYTRACING);
  cmd_context->CmdBindResources(1, {film->accumulated_samples_.get()}, graphics::BIND_POINT_RAYTRACING);
  cmd_context->CmdDispatchRays(film->accumulated_color_->Extent().width, film->accumulated_samples_->Extent().height,
                               1);
  core_->GraphicsCore()->SubmitCommandContext(cmd_context.get());
  core_->GraphicsCore()->WaitGPU();
}

void Scene::UpdatePipeline() {
  rt_program_.reset();
  raygen_shader_.reset();
  tlas_.reset();
  core_->GraphicsCore()->CreateRayTracingProgram(&rt_program_);
  core_->GraphicsCore()->CreateShader(core_->GetShadersVFS(), "raygen.hlsl", "Main", "lib_6_3", &raygen_shader_);
  // core_->GraphicsCore()->CreateTopLevelAccelerationStructure(std::vector<graphics::RayTracingInstance>{}, &tlas_);
  rt_program_->AddResourceBinding(graphics::RESOURCE_TYPE_IMAGE, 1);
  rt_program_->AddResourceBinding(graphics::RESOURCE_TYPE_IMAGE, 1);
  rt_program_->AddRayGenShader(raygen_shader_.get());
  rt_program_->Finalize({}, {}, {});
}

}  // namespace sparks
