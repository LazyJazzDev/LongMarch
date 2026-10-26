#include "sparkium/pipelines/raster/entity/entity_geometry_material.h"

#include "glm/gtc/matrix_transform.hpp"
#include "sparkium/pipelines/raster/core/camera.h"
#include "sparkium/pipelines/raster/core/core.h"
#include "sparkium/pipelines/raster/core/film.h"
#include "sparkium/pipelines/raster/core/scene.h"
#include "sparkium/pipelines/raster/geometry/geometries.h"
#include "sparkium/pipelines/raster/material/materials.h"

namespace sparkium::raster {

EntityGeometryMaterial::EntityGeometryMaterial(sparkium::EntityGeometryMaterial &entity)
    : entity_(entity),
      Entity(DedicatedCast(entity.GetCore())),
      geometry_(DedicatedCast(entity_.GetGeometry())),
      material_(DedicatedCast(entity_.GetMaterial())) {
  instance_buffer_ = core_->GetDevice()->CreateBuffer({(sizeof(InstanceData) + 255) & ~uint64_t(255)});
  rhi::ProgramDesc render_program_desc;
  render_program_desc.color_formats = {rhi::IMAGE_FORMAT_R32G32B32A32_SFLOAT, rhi::IMAGE_FORMAT_R32G32B32A32_SFLOAT,
                                       rhi::IMAGE_FORMAT_R32G32B32A32_SFLOAT, rhi::IMAGE_FORMAT_R32G32B32A32_SFLOAT,
                                       rhi::IMAGE_FORMAT_R32_INT};
  render_program_desc.depth_stencil = {rhi::IMAGE_FORMAT_D32_SFLOAT, true, true};
  render_program_desc.pipeline_layout.groups.resize(1);
  render_program_desc.vertex_shader = geometry_->VertexShader();
  render_program_desc.fragment_shader = material_->PixelShader();
  geometry_->SetupProgram(render_program_desc);
  render_program_desc.pipeline_layout.groups[0].bindings.push_back({0, rhi::BINDING_UNIFORM_BUFFER, 1});  // Camera Data
  render_program_desc.pipeline_layout.groups[0].bindings.push_back(
      {1, rhi::BINDING_UNIFORM_BUFFER, 1});  // Instance Data
  render_program_desc.pipeline_layout.groups[0].bindings.push_back(
      {2, rhi::BINDING_BYTE_ADDRESS_BUFFER, 1});  // Material Data
  material_->SetupProgram(render_program_desc);
  render_program_ = core_->GetDevice()->CreateProgram(render_program_desc);
  render_group_ = core_->GetDevice()->CreateBindingGroup(render_program_desc.pipeline_layout.groups[0]);

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

void EntityGeometryMaterial::Update(Scene *scene) {
  instance_data.model[0] = {entity_.transform[0], 0.0f};
  instance_data.model[1] = {entity_.transform[1], 0.0f};
  instance_data.model[2] = {entity_.transform[2], 0.0f};
  instance_data.model[3] = {entity_.transform[3], 1.0f};
  instance_data.inv_model = glm::inverse(instance_data.model);
  instance_data.normal_matrix = glm::transpose(instance_data.inv_model);
  UploadBuffer(core_->GetDevice(), instance_buffer_.Get(), &instance_data, sizeof(InstanceData));
  material_->Sync();

  scene->RegisterRenderCallback([this](rhi::ICommandContext *cmd_ctx, rhi::IBuffer *camera_buffer) {
    cmd_ctx->CmdBindProgram(render_program_.Get());
    render_group_->BindResource(0, camera_buffer);
    render_group_->BindResource(1, instance_buffer_.Get());
    material_->BindMaterialResources(render_group_.Get());
    cmd_ctx->CmdBindGroup(0, render_group_.Get());
    geometry_->DispatchDrawCalls(cmd_ctx);
  });

  if (entity_.raster_light) {
    glm::vec3 emission = material_->Emission();
    if (emission.r > 0.0f || emission.g > 0.0f || emission.b > 0.0f) {
      glm::vec4 centric_area = geometry_->CentricArea(entity_.transform);
      point_light_data.emission = emission * glm::max(0.0f, centric_area.w) * 4.0f * glm::pi<float>();
      point_light_data.position = {centric_area.x, centric_area.y, centric_area.z};
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
  }
}

}  // namespace sparkium::raster
