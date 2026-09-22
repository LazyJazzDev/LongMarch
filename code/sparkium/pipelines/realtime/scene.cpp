#include "sparkium/pipelines/realtime/scene.h"

#include "grassland/graphics/frame_profile.h"
#include "sparkium/pipelines/common/core/camera.h"
#include "sparkium/pipelines/common/core/core.h"
#include "sparkium/pipelines/realtime/realtime_view.h"

namespace sparkium::realtime {
namespace {
RealtimeView *View(sparkium::Film *film) {
  COMPONENT_CAST(film, RealtimeView);
}
}  // namespace

Scene::Scene(sparkium::Scene &scene) : render_shared::Scene(scene) {
  software_tracing_ = true;
  render_shared::ComputeShadingConfiguration shading;
  shading.source = "realtime/trace.hlsl";
  shading.definitions = {"-DSPARKIUM_REALTIME"};
  shading.output_images = 3;
  shading.input_type = graphics::RESOURCE_TYPE_IMAGE;
  shading.input_images = 4;
  shading.extra_buffers = 1;
  shading.auxiliary_entry = "Reproject";
  software_pipeline_ = std::make_unique<render_shared::SoftwarePipeline>(core_, false, std::move(shading));
}

void Scene::Render(sparkium::Camera *source_camera, sparkium::Film *film) {
  auto *core = core_;
  auto *camera = render_shared::DedicatedCast(source_camera);
  auto *view = View(film);
  auto *pipeline = software_pipeline_.get();
  const auto &settings = scene_.settings;
  graphics::CpuProfileScope update_profile("scene_update");
  const uint64_t key = RealtimeKey();
  const bool scene_changed = !rendered_ || realtime_key_ != key || pipeline_dirty_;
  if (scene_changed || film->info.accumulated_samples == 0) {
    UpdatePipeline(camera);
    if (scene_changed)
      film->Reset();
    realtime_key_ = key;
  }
  update_profile.End();
  graphics::CpuProfileScope setup_profile("render_setup");
  rendered_ = true;
  auto trace_settings = settings.raytracing;
  trace_settings.samples_per_dispatch = 1;
  trace_settings.max_bounces = std::clamp(settings.realtime.bounces, 1, 8);
  const uint32_t frame = film->info.accumulated_samples;
  scene_settings_buffer_->UploadData(&trace_settings, sizeof(trace_settings));
  scene_settings_buffer_->UploadData(&film->info, sizeof(sparkium::Film::Info), sizeof(trace_settings));
  ++film->info.accumulated_samples;
  std::unique_ptr<graphics::CommandContext> commands;
  core->GraphicsCore()->CreateCommandContext(&commands);
  view->Begin(commands.get(), pipeline, buffers_, camera, film->GetWidth(), film->GetHeight(), settings.realtime.scale,
              settings.realtime.history, settings.realtime.updates, frame);
  graphics::GpuProfileScope trace_profile(commands.get(), "realtime_lighting");
  for (int phase = 0; phase < 2; ++phase) {
    commands->CmdBindComputeProgram(phase == 0 ? pipeline->AuxiliaryProgram() : pipeline->Program());
    constexpr auto bind_point = graphics::BIND_POINT_COMPUTE;
    commands->CmdBindResources(2, {pipeline->Nodes()}, bind_point);
    commands->CmdBindResources(3, {scene_settings_buffer_.get()}, bind_point);
    auto resources = buffers_;
    resources.insert(resources.end(), {core->GetBuffer("sobol"), camera->Buffer(), instance_metadata_buffer_.get(),
                                       light_selector_buffer_.get(), light_metadatas_buffer_.get(),
                                       pipeline->Instances(), view->ParametersBuffer()});
    commands->CmdBindResources(4, resources, bind_point);
    commands->CmdBindResources(5, sdr_images_, bind_point);
    commands->CmdBindResources(6, hdr_images_, bind_point);
    commands->CmdBindResources(7, std::vector{linear_sampler_.get(), nearest_sampler_.get()}, bind_point);
    view->BindTrace(commands.get());
    uint32_t trace_width = view->Width();
    if (phase == 1) {
      const uint32_t updates = std::clamp(settings.realtime.updates, 1, 16);
      trace_width = (trace_width + updates - 1) / updates;
    }
    commands->CmdDispatch((trace_width + 7) / 8, (view->Height() + 7) / 8, 1);
  }
  trace_profile.End();
  graphics::GpuProfileScope resolve_profile(commands.get(), "film_resolve");
  view->Resolve(commands.get(), film->GetRawImage(), pipeline, buffers_);
  resolve_profile.End();
  setup_profile.End();
  graphics::CpuProfileScope submit_profile("render_submit");
  core->GraphicsCore()->SubmitCommandContext(commands.get());
  submit_profile.End();
  graphics::CpuProfileScope wait_profile("render_wait");
  core->GraphicsCore()->WaitGPU();
}

uint64_t Scene::RealtimeKey() const {
  uint64_t key = 1469598103934665603ull;
  auto add = [&](const auto &value) {
    const auto *bytes = reinterpret_cast<const uint8_t *>(&value);
    for (size_t i = 0; i < sizeof(value); ++i)
      key = (key ^ bytes[i]) * 1099511628211ull;
  };
  add(scene_.settings.raytracing);
  add(scene_.settings.realtime);
  for (auto *entity : scene_.GetEntityOrder()) {
    add(entity);
    add(scene_.GetEntities().at(entity).active);
    if (auto *instance = dynamic_cast<sparkium::EntityGeometryMaterial *>(entity)) {
      add(instance->transform);
      auto *geometry = instance->GetGeometry();
      auto *material = instance->GetMaterial();
      add(geometry);
      add(material);
      if (auto *m = dynamic_cast<sparkium::MaterialLambertian *>(material)) {
        add(m->base_color);
        add(m->emission);
      }
      if (auto *m = dynamic_cast<sparkium::MaterialPrincipled *>(material)) {
        add(m->info);
        add(m->textures);
      }
      if (auto *m = dynamic_cast<sparkium::MaterialSpecular *>(material))
        add(m->base_color);
      if (auto *m = dynamic_cast<sparkium::MaterialLight *>(material)) {
        add(m->emission);
        add(m->block_ray);
        add(m->camera_visible);
        add(m->two_sided);
        add(m->falloff_distance);
      }
      if (auto *m = dynamic_cast<sparkium::MaterialShaderGraph *>(material))
        add(m->emission_hint);
    }
    if (auto *light = dynamic_cast<sparkium::EntityPointLight *>(entity)) {
      add(light->position);
      add(light->color);
      add(light->strength);
      add(light->radius);
      add(light->soft_falloff);
      add(light->sampling_weight);
    }
  }
  return key;
}

}  // namespace sparkium::realtime
