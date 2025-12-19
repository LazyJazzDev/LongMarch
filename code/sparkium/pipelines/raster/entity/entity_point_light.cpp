#include "sparkium/pipelines/raster/entity/entity_point_light.h"

#include "sparkium/pipelines/raster/core/camera.h"
#include "sparkium/pipelines/raster/core/core.h"
#include "sparkium/pipelines/raster/core/film.h"
#include "sparkium/pipelines/raster/core/scene.h"

namespace sparkium::raster {

EntityPointLight::EntityPointLight(sparkium::EntityPointLight &entity)
    : entity_(entity), Entity(DedicatedCast(entity.GetCore())) {
  core_->GraphicsCore()->CreateBuffer(sizeof(PointLightData), graphics::BUFFER_TYPE_DYNAMIC, &point_light_buffer_);
  core_->GraphicsCore()->CreateShader(core_->GetShadersVFS(), "light/point/lighting.hlsl", "VSMain", "vs_6_0", {"-I."},
                                      &point_light_vs_);
  core_->GraphicsCore()->CreateShader(core_->GetShadersVFS(), "light/point/lighting.hlsl", "PSMain", "ps_6_0", {"-I."},
                                      &point_light_ps_);
  core_->GraphicsCore()->CreateProgram({graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT}, graphics::IMAGE_FORMAT_UNDEFINED,
                                       &point_light_program_);
  point_light_program_->BindShader(point_light_vs_.get(), graphics::SHADER_TYPE_VERTEX);
  point_light_program_->BindShader(point_light_ps_.get(), graphics::SHADER_TYPE_PIXEL);
  point_light_program_->SetBlendState(
      0, graphics::BlendState{graphics::BLEND_FACTOR_ONE, graphics::BLEND_FACTOR_ONE, graphics::BLEND_OP_ADD,
                              graphics::BLEND_FACTOR_ONE, graphics::BLEND_FACTOR_ONE, graphics::BLEND_OP_ADD});
  point_light_program_->AddResourceBinding(graphics::RESOURCE_TYPE_IMAGE, 1);           // AlbedoRoughness
  point_light_program_->AddResourceBinding(graphics::RESOURCE_TYPE_IMAGE, 1);           // PositionSpecular
  point_light_program_->AddResourceBinding(graphics::RESOURCE_TYPE_IMAGE, 1);           // NormalMetallic
  point_light_program_->AddResourceBinding(graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);  // Camera Data
  point_light_program_->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, 1);  // PointLightData
  point_light_program_->Finalize();
}

void EntityPointLight::Update(Scene *scene) {
  point_light_data.position = entity_.position;
  point_light_data.emission = entity_.color * entity_.strength;
  point_light_buffer_->UploadData(&point_light_data, sizeof(PointLightData));
  scene->RegisterLightingCallback([this](graphics::CommandContext *cmd_ctx, Camera *camera, Film *film) {
    cmd_ctx->CmdBindProgram(point_light_program_.get());
    cmd_ctx->CmdBindResources(0, {film->GetAlbedoRoughnessBuffer()}, graphics::BIND_POINT_GRAPHICS);
    cmd_ctx->CmdBindResources(1, {film->GetPositionSpecularBuffer()}, graphics::BIND_POINT_GRAPHICS);
    cmd_ctx->CmdBindResources(2, {film->GetNormalMetallicBuffer()}, graphics::BIND_POINT_GRAPHICS);
    cmd_ctx->CmdBindResources(3, {camera->NearFieldBuffer()}, graphics::BIND_POINT_GRAPHICS);
    cmd_ctx->CmdBindResources(4, {point_light_buffer_.get()}, graphics::BIND_POINT_GRAPHICS);
    cmd_ctx->CmdDraw(6, 1, 0, 0);
  });
}

}  // namespace sparkium::raster
