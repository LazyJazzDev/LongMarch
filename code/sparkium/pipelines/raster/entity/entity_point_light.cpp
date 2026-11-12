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
  auto shaders_vfs = core_->GetShadersVFS();
  point_light_vs_ = core_->GetDevice()->CreateShader("light/point/lighting.slang", "VSMain", rhi::SHADER_STAGE_VERTEX,
                                                     &shaders_vfs, {"."}, {});
  point_light_ps_ = core_->GetDevice()->CreateShader("light/point/lighting.slang", "PSMain", rhi::SHADER_STAGE_FRAGMENT,
                                                     &shaders_vfs, {"."}, {});
  rhi::ProgramDesc point_light_program_desc;
  point_light_program_desc.color_formats = {rhi::IMAGE_FORMAT_R32G32B32A32_SFLOAT};
  point_light_program_desc.pipeline_layout.groups.resize(3);
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
  point_light_program_desc.pipeline_layout.groups[2].bindings.push_back(
      {0, rhi::BINDING_UNIFORM_BUFFER, 1});  // Camera Data
  point_light_program_desc.pipeline_layout.groups[1].bindings.push_back(
      {4, rhi::BINDING_BYTE_ADDRESS_BUFFER, 1});  // PointLightData
  point_light_program_ = core_->GetDevice()->CreateProgram(point_light_program_desc);
  point_light_group_ = core_->GetDevice()->CreateBindingGroup(point_light_program_desc.pipeline_layout.groups[1]);
}

void EntityPointLight::Update(Scene *scene) {
  point_light_data.position = entity_.position;
  point_light_data.emission = entity_.color * entity_.strength;
  core_->QueueUpload(point_light_buffer_.Get(), &point_light_data, sizeof(PointLightData));
  scene->RegisterLightingCallback([this](rhi::ICommandContext *cmd_ctx, Camera *camera, Film *film) {
    cmd_ctx->CmdBindProgram(point_light_program_.Get());

    cmd_ctx->CmdBindGroup(2, camera->NearFieldBindings());
    point_light_group_->BindResource(4, point_light_buffer_.Get());
    cmd_ctx->CmdBindGroup(0, film->LightingBindings());
    cmd_ctx->CmdBindGroup(1, point_light_group_.Get());
    cmd_ctx->CmdDraw(6, 1, 0, 0);
  });
}

}  // namespace sparkium::raster
