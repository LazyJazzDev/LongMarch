#include "sparkium/pipelines/raster/entity/entity_point_light.h"

#include "sparkium/pipelines/raster/core/camera.h"
#include "sparkium/pipelines/raster/core/core.h"
#include "sparkium/pipelines/raster/core/film.h"
#include "sparkium/pipelines/raster/core/scene.h"

namespace sparkium::raster {

EntityPointLight::EntityPointLight(sparkium::EntityPointLight &entity)
    : entity_(entity),
      Entity(DedicatedCast(entity.GetCore())) {
  point_light_buffer_ = core_->GetDevice()->CreateBuffer({(sizeof(PointLightData) + 255) & ~uint64_t(255)});
  point_light_vs_ = core_->GetDevice()->CreateShader("light/point/lighting.slang", "VSMain", rhi::SHADER_STAGE_VERTEX,
                                                     &core_->GetShadersVFS(), {"."}, {});
  point_light_ps_ = core_->GetDevice()->CreateShader("light/point/lighting.slang", "PSMain", rhi::SHADER_STAGE_FRAGMENT,
                                                     &core_->GetShadersVFS(), {"."}, {});
  rhi::ProgramDesc point_light_program_desc;
  point_light_program_desc.color_formats = {rhi::IMAGE_FORMAT_R32G32B32A32_SFLOAT};
  point_light_program_desc.pipeline_layout.groups.resize(1);
  point_light_program_desc.vertex_shader = point_light_vs_.Get();
  point_light_program_desc.fragment_shader = point_light_ps_.Get();
  point_light_program_desc.blend_states = {rhi::BlendState{rhi::BLEND_FACTOR_ONE, rhi::BLEND_FACTOR_ONE,
                                                           rhi::BLEND_OP_ADD, rhi::BLEND_FACTOR_ONE,
                                                           rhi::BLEND_FACTOR_ONE, rhi::BLEND_OP_ADD}};
  point_light_program_desc.pipeline_layout.groups[0].bindings.push_back(
      {0, rhi::BINDING_TEXTURE2D, 1});  // AlbedoRoughness
  point_light_program_desc.pipeline_layout.groups[0].bindings.push_back(
      {1, rhi::BINDING_TEXTURE2D, 1});  // PositionSpecular
  point_light_program_desc.pipeline_layout.groups[0].bindings.push_back(
      {2, rhi::BINDING_TEXTURE2D, 1});  // NormalMetallic
  point_light_program_desc.pipeline_layout.groups[0].bindings.push_back(
      {3, rhi::BINDING_UNIFORM_BUFFER, 1});  // Camera Data
  point_light_program_desc.pipeline_layout.groups[0].bindings.push_back(
      {4, rhi::BINDING_BYTE_ADDRESS_BUFFER, 1});  // PointLightData
  point_light_program_ = core_->GetDevice()->CreateProgram(point_light_program_desc);
  point_light_group_ = core_->GetDevice()->CreateBindingGroup(point_light_program_desc.pipeline_layout.groups[0]);
}

void EntityPointLight::Update(Scene *scene) {
  point_light_data.position = entity_.position;
  point_light_data.emission = entity_.color * entity_.strength;
  UploadBuffer(core_->GetDevice(), point_light_buffer_.Get(), &point_light_data, sizeof(PointLightData));
  scene->RegisterLightingCallback([this](rhi::ICommandContext *cmd_ctx, Camera *camera, Film *film) {
    cmd_ctx->CmdBindProgram(point_light_program_.Get());
    point_light_group_->BindResource(0, film->GetAlbedoRoughnessBuffer());
    point_light_group_->BindResource(1, film->GetPositionSpecularBuffer());
    point_light_group_->BindResource(2, film->GetNormalMetallicBuffer());
    point_light_group_->BindResource(3, camera->NearFieldBuffer());
    point_light_group_->BindResource(4, point_light_buffer_.Get());
    cmd_ctx->CmdBindGroup(0, point_light_group_.Get());
    cmd_ctx->CmdDraw(6, 1, 0, 0);
  });
}

}  // namespace sparkium::raster
