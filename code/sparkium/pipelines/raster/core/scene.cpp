#include "sparkium/core/scene.h"

#include "sparkium/core/film.h"
#include "sparkium/pipelines/raster/core/camera.h"
#include "sparkium/pipelines/raster/core/core.h"
#include "sparkium/pipelines/raster/core/film.h"
#include "sparkium/pipelines/raster/core/scene.h"
#include "sparkium/pipelines/raster/entity/entities.h"

namespace sparkium::raster {

Scene::Scene(sparkium::Scene &scene) : scene_(scene), core_(DedicatedCast(scene.GetCore())), settings(scene.settings) {
  core_->GraphicsCore()->CreateShader(core_->GetShadersVFS(), "light/ambient/lighting.hlsl", "VSMain", "vs_6_0", {},
                                      &ambient_light_vs_);
  core_->GraphicsCore()->CreateShader(core_->GetShadersVFS(), "light/ambient/lighting.hlsl", "PSMain", "ps_6_0", {},
                                      &ambient_light_ps_);
  core_->GraphicsCore()->CreateProgram({graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT}, graphics::IMAGE_FORMAT_UNDEFINED,
                                       &ambient_light_program_);
  ambient_light_program_->BindShader(ambient_light_vs_.get(), graphics::SHADER_TYPE_VERTEX);
  ambient_light_program_->BindShader(ambient_light_ps_.get(), graphics::SHADER_TYPE_PIXEL);
  ambient_light_program_->SetBlendState(
      0, graphics::BlendState{graphics::BLEND_FACTOR_ONE, graphics::BLEND_FACTOR_ONE, graphics::BLEND_OP_ADD,
                              graphics::BLEND_FACTOR_ONE, graphics::BLEND_FACTOR_ONE, graphics::BLEND_OP_ADD});
  ambient_light_program_->AddResourceBinding(graphics::RESOURCE_TYPE_IMAGE, 1);           // Camera Data
  ambient_light_program_->AddResourceBinding(graphics::RESOURCE_TYPE_IMAGE, 1);           // Camera Data
  ambient_light_program_->AddResourceBinding(graphics::RESOURCE_TYPE_IMAGE, 1);           // Camera Data
  ambient_light_program_->AddResourceBinding(graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);  // Camera Data
  ambient_light_program_->Finalize();

  core_->GraphicsCore()->CreateBuffer(sizeof(glm::vec3), graphics::BUFFER_TYPE_DYNAMIC, &ambient_light_buffer_);
}

void Scene::Render(Camera *camera, Film *film) {
  for (auto &[entity, status] : entities_) {
    status.keep = false;
  }
  for (auto [entity, status] : scene_.GetEntities()) {
    auto actual_entity = DedicatedCast(entity);
    if (actual_entity) {
      entities_[actual_entity].keep = true;
      entities_[actual_entity].active = status.active;
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

  render_callbacks_.clear();
  shadow_map_callbacks_.clear();
  lighting_callbacks_.clear();

  ambient_light_buffer_->UploadData(&settings.ambient_light, sizeof(glm::vec3));

  for (auto &[entity, status] : entities_) {
    if (status.active) {
      entity->Update(this);
    }
  }

  camera->Update();

  film->film_.Reset();

  std::unique_ptr<graphics::CommandContext> cmd_context;
  core_->GraphicsCore()->CreateCommandContext(&cmd_context);

  // Render Pass

  cmd_context->CmdClearImage(film->film_.GetDepthImage(), {1.0f, 0.0f, 0.0f, 0.0f});
  cmd_context->CmdBeginRendering(
      {film->film_.GetRawImage(), film->albedo_roughness_buffer_.get(), film->position_specular_buffer_.get(),
       film->normal_metallic_buffer_.get(), film->film_.GetStencilImage()},
      film->film_.GetDepthImage());
  cmd_context->CmdSetPrimitiveTopology(graphics::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
  graphics::Scissor scissor;
  scissor.offset = {0, 0};
  scissor.extent = film->film_.GetExtent();
  graphics::Viewport viewport;
  viewport.x = 0.0f;
  viewport.y = 0.0f;
  viewport.min_depth = 0.0f;
  viewport.max_depth = 1.0f;
  viewport.width = static_cast<float>(film->film_.GetWidth());
  viewport.height = static_cast<float>(film->film_.GetHeight());
  cmd_context->CmdSetScissor(scissor);
  cmd_context->CmdSetViewport(viewport);
  for (auto &callback : render_callbacks_) {
    callback(cmd_context.get(), camera->FarFieldBuffer());
  }
  cmd_context->CmdEndRendering();
  cmd_context->CmdClearImage(film->film_.GetDepthImage(), {1.0f, 0.0f, 0.0f, 0.0f});
  cmd_context->CmdBeginRendering(
      {film->film_.GetRawImage(), film->albedo_roughness_buffer_.get(), film->position_specular_buffer_.get(),
       film->normal_metallic_buffer_.get(), film->film_.GetStencilImage()},
      film->film_.GetDepthImage());
  cmd_context->CmdSetPrimitiveTopology(graphics::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
  cmd_context->CmdSetScissor(scissor);
  cmd_context->CmdSetViewport(viewport);
  for (auto &callback : render_callbacks_) {
    callback(cmd_context.get(), camera->NearFieldBuffer());
  }
  cmd_context->CmdSetScissor(scissor);
  cmd_context->CmdSetViewport(viewport);
  cmd_context->CmdEndRendering();

  // Lighting Pass
  cmd_context->CmdBeginRendering({film->film_.GetRawImage()}, nullptr);
  cmd_context->CmdSetPrimitiveTopology(graphics::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
  cmd_context->CmdSetScissor(scissor);
  cmd_context->CmdSetViewport(viewport);
  cmd_context->CmdBindProgram(ambient_light_program_.get());
  cmd_context->CmdBindResources(0, {film->albedo_roughness_buffer_.get()}, graphics::BIND_POINT_GRAPHICS);
  cmd_context->CmdBindResources(1, {film->position_specular_buffer_.get()}, graphics::BIND_POINT_GRAPHICS);
  cmd_context->CmdBindResources(2, {film->normal_metallic_buffer_.get()}, graphics::BIND_POINT_GRAPHICS);
  cmd_context->CmdBindResources(3, {ambient_light_buffer_.get()}, graphics::BIND_POINT_GRAPHICS);
  cmd_context->CmdDraw(6, 1, 0, 0);

  for (auto &callback : lighting_callbacks_) {
    callback(cmd_context.get());
  }

  cmd_context->CmdEndRendering();

  core_->GraphicsCore()->SubmitCommandContext(cmd_context.get());
  core_->GraphicsCore()->WaitGPU();
}

void Scene::RegisterRenderCallback(
    const std::function<void(graphics::CommandContext *, graphics::Buffer *)> &callback) {
  render_callbacks_.push_back(callback);
}

void Scene::RegisterShadowMapCallback(const std::function<void(graphics::CommandContext *)> &callback) {
  shadow_map_callbacks_.push_back(callback);
}

void Scene::RegisterLightingCallback(const std::function<void(graphics::CommandContext *)> &callback) {
  lighting_callbacks_.push_back(callback);
}

Scene *DedicatedCast(sparkium::Scene *scene) {
  COMPONENT_CAST(scene, Scene);
}

}  // namespace sparkium::raster
