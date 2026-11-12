#include "sparkium/core/scene.h"

#include "sparkium/core/film.h"
#include "sparkium/core/renderer.h"
#include "sparkium/pipelines/raster/core/camera.h"
#include "sparkium/pipelines/raster/core/core.h"
#include "sparkium/pipelines/raster/core/film.h"
#include "sparkium/pipelines/raster/core/scene.h"
#include "sparkium/pipelines/raster/entity/entities.h"

namespace sparkium::raster {

Scene::Scene(sparkium::Scene &scene) : scene_(scene), core_(DedicatedCast(scene.GetCore())), settings(scene.settings) {
  auto shaders_vfs = core_->GetShadersVFS();
  ambient_light_vs_ = core_->GetDevice()->CreateShader("light/ambient/lighting.slang", "VSMain",
                                                       rhi::SHADER_STAGE_VERTEX, &shaders_vfs, {"."}, {});
  ambient_light_ps_ = core_->GetDevice()->CreateShader("light/ambient/lighting.slang", "PSMain",
                                                       rhi::SHADER_STAGE_FRAGMENT, &shaders_vfs, {"."}, {});
  rhi::ProgramDesc ambient_light_program_desc;
  ambient_light_program_desc.color_formats = {rhi::IMAGE_FORMAT_R32G32B32A32_SFLOAT};
  ambient_light_program_desc.pipeline_layout.groups.resize(3);
  ambient_light_program_desc.vertex_shader = ambient_light_vs_.Get();
  ambient_light_program_desc.fragment_shader = ambient_light_ps_.Get();
  ambient_light_program_desc.blend_states = {rhi::BlendState{rhi::BLEND_FACTOR_ONE, rhi::BLEND_FACTOR_ONE,
                                                             rhi::BLEND_OP_ADD, rhi::BLEND_FACTOR_ONE,
                                                             rhi::BLEND_FACTOR_ONE, rhi::BLEND_OP_ADD}};
  ambient_light_program_desc.pipeline_layout.groups[0].bindings.push_back(
      {0, rhi::BINDING_TEXTURE2D, 1});  // AlbedoRoughness
  ambient_light_program_desc.pipeline_layout.groups[0].bindings.push_back(
      {1, rhi::BINDING_TEXTURE2D, 1});  // PositionSpecular
  ambient_light_program_desc.pipeline_layout.groups[0].bindings.push_back(
      {2, rhi::BINDING_TEXTURE2D, 1});  // NormalMetallic
  ambient_light_program_desc.pipeline_layout.groups[2].bindings.push_back(
      {0, rhi::BINDING_UNIFORM_BUFFER, 1});  // Camera Data
  ambient_light_program_desc.pipeline_layout.groups[1].bindings.push_back(
      {4, rhi::BINDING_UNIFORM_BUFFER, 1});  // Ambient Light Data
  ambient_light_description_ = std::make_shared<const rhi::ProgramDesc>(ambient_light_program_desc);
  ambient_light_group_ = core_->GetDevice()->CreateBindingGroup(ambient_light_program_desc.pipeline_layout.groups[1]);

  ambient_light_buffer_ = core_->GetDevice()->CreateBuffer({(sizeof(glm::vec3) + 255) & ~uint64_t(255)});
  if (!(ambient_light_vs_ && ambient_light_ps_ && ambient_light_group_ && ambient_light_buffer_))
    core_->ReportFailure("Cannot create raster resources");
}

bool Scene::RecordRender(sparkium::Renderer &renderer,
                         rhi::ICommandContext *commands,
                         ResourceUpdates::Cursor &updates,
                         Camera *camera,
                         Film *film) {
  for (auto &[entity, status] : entities_) {
    status.keep = false;
  }

  std::vector<Entity *> ordered_entities;
  for (auto *entity : scene_.GetEntityOrder()) {
    const auto &status = scene_.GetEntities().at(entity);
    auto actual_entity = DedicatedCast(entity);
    if (actual_entity) {
      entities_[actual_entity].keep = true;
      entities_[actual_entity].active = status.active;
      ordered_entities.push_back(actual_entity);
    }
  }

  std::vector<Entity *> to_remove;
  for (auto &[entity, status] : entities_) {
    if (!status.keep) {
      to_remove.push_back(entity);
    }
  }

  for (auto entity : to_remove) {
    entities_.erase(entity);
  }

  if (!core_->IsValid())
    return false;

  render_callbacks_.clear();
  lighting_callbacks_.clear();

  core_->QueueUpload(ambient_light_buffer_.Get(), &settings.ambient_light, sizeof(glm::vec3));

  for (auto *entity : ordered_entities) {
    if (entities_.at(entity).active) {
      entity->Update(this);
    }
  }

  camera->Update();

  if (!core_->RecordUpdates(commands, updates) || !film->RecordReset(commands))
    return false;

  // Render Pass

  if (!commands->CmdClearImage(
          film->film_.GetRawImage(),
          {scene_.settings.ambient_light.r, scene_.settings.ambient_light.g, scene_.settings.ambient_light.b, 0.0f}))
    return false;
  if (!commands->CmdClearImage(film->film_.GetDepthImage(), rhi::ClearValue{rhi::ClearDepthStencilValue{1.0f, 0}}))
    return false;
  if (!commands->CmdClearImage(film->albedo_roughness_buffer_.Get(), {}))
    return false;
  if (!commands->CmdClearImage(film->position_specular_buffer_.Get(), {}))
    return false;
  if (!commands->CmdClearImage(film->normal_metallic_buffer_.Get(), {}))
    return false;
  if (!commands->CmdClearImage(film->film_.GetStencilImage(), rhi::ClearValue{0, 0, 0, 0}))
    return false;
  rhi::RenderingInfo rendering;
  rendering.color_attachments = {{film->film_.GetRawImage()},
                                 {film->albedo_roughness_buffer_.Get()},
                                 {film->position_specular_buffer_.Get()},
                                 {film->normal_metallic_buffer_.Get()},
                                 {film->film_.GetStencilImage()}};
  rendering.depth_stencil.image = film->film_.GetDepthImage();
  if (!commands->CmdBeginRendering(rendering))
    return false;
  rhi::Scissor scissor{0, 0, static_cast<uint32_t>(film->film_.GetWidth()),
                       static_cast<uint32_t>(film->film_.GetHeight())};
  rhi::Viewport viewport;
  viewport.x = 0.0f;
  viewport.y = 0.0f;
  viewport.min_depth = 0.0f;
  viewport.max_depth = 1.0f;
  viewport.width = static_cast<float>(film->film_.GetWidth());
  viewport.height = static_cast<float>(film->film_.GetHeight());
  if (!commands->CmdSetScissor(scissor))
    return false;
  if (!commands->CmdSetViewport(viewport))
    return false;
  for (auto &callback : render_callbacks_) {
    if (!callback(renderer, commands, camera->FarFieldBindings()))
      return false;
  }
  if (!commands->CmdEndRendering())
    return false;
  // Clear depth to 1.0 for the near field rendering.
  if (!commands->CmdClearImage(film->film_.GetDepthImage(), rhi::ClearValue{rhi::ClearDepthStencilValue{1.0f, 0}}))
    return false;
  if (!commands->CmdBeginRendering(rendering))
    return false;
  if (!commands->CmdSetScissor(scissor))
    return false;
  if (!commands->CmdSetViewport(viewport))
    return false;
  for (auto &callback : render_callbacks_) {
    if (!callback(renderer, commands, camera->NearFieldBindings()))
      return false;
  }
  if (!commands->CmdEndRendering())
    return false;

  // Lighting Pass
  rendering.color_attachments = {{film->film_.GetRawImage()}};
  rendering.depth_stencil = {};
  if (!commands->CmdBeginRendering(rendering))
    return false;
  if (!commands->CmdSetScissor(scissor))
    return false;
  if (!commands->CmdSetViewport(viewport))
    return false;
  if (!renderer.RecordProgram(commands, ambient_light_description_))
    return false;

  if (!commands->CmdBindGroup(2, camera->NearFieldBindings()))
    return false;
  if (!ambient_light_group_ || !ambient_light_group_->BindResource(4, ambient_light_buffer_.Get()))
    return false;
  if (!commands->CmdBindGroup(0, film->LightingBindings()))
    return false;
  if (!commands->CmdBindGroup(1, ambient_light_group_.Get()))
    return false;
  if (!commands->CmdDraw(6, 1, 0, 0))
    return false;

  for (auto &callback : lighting_callbacks_) {
    if (!callback(renderer, commands, camera, film))
      return false;
  }

  if (!commands->CmdEndRendering())
    return false;

  return core_->IsValid();
}

void Scene::RegisterRenderCallback(
    const std::function<bool(sparkium::Renderer &, rhi::ICommandContext *, rhi::IBindingGroup *)> &callback) {
  render_callbacks_.push_back(callback);
}

void Scene::RegisterLightingCallback(
    const std::function<bool(sparkium::Renderer &, rhi::ICommandContext *, Camera *camera, Film *film)> &callback) {
  lighting_callbacks_.push_back(callback);
}

Scene *DedicatedCast(sparkium::Scene *scene) {
  COMPONENT_CAST(scene, Scene);
}

}  // namespace sparkium::raster
