#include "sparkium/pipelines/raster/entity/entity_geometry_material.h"

#include "glm/gtc/matrix_transform.hpp"
#include "sparkium/pipelines/raster/core/core.h"
#include "sparkium/pipelines/raster/core/scene.h"
#include "sparkium/pipelines/raster/geometry/geometries.h"
#include "sparkium/pipelines/raster/material/materials.h"

namespace sparkium::raster {

EntityGeometryMaterial::EntityGeometryMaterial(sparkium::EntityGeometryMaterial &entity)
    : entity_(entity),
      Entity(DedicatedCast(entity.GetCore())),
      geometry_(DedicatedCast(entity_.GetGeometry())),
      material_(DedicatedCast(entity_.GetMaterial())) {
  core_->GraphicsCore()->CreateBuffer(sizeof(InstanceData), graphics::BUFFER_TYPE_DYNAMIC, &instance_buffer_);
  core_->GraphicsCore()->CreateProgram(
      {graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT, graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT,
       graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT, graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT,
       graphics::IMAGE_FORMAT_R32_SINT},
      graphics::IMAGE_FORMAT_D32_SFLOAT, &render_program_);
  render_program_->BindShader(geometry_->VertexShader(), graphics::SHADER_TYPE_VERTEX);
  render_program_->BindShader(material_->PixelShader(), graphics::SHADER_TYPE_PIXEL);
  geometry_->SetupProgram(render_program_.get());
  render_program_->AddResourceBinding(graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);  // Camera Data
  render_program_->AddResourceBinding(graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);  // Instance Data
  render_program_->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, 1);  // Material Data
  material_->SetupProgram(render_program_.get());
  render_program_->Finalize();

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
  point_light_program_->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, 1);  // PointLightData
  point_light_program_->Finalize();
}

void EntityGeometryMaterial::Update(Scene *scene) {
  instance_data.model[0] = {entity_.transform[0], 0.0f};
  instance_data.model[1] = {entity_.transform[1], 0.0f};
  instance_data.model[2] = {entity_.transform[2], 0.0f};
  instance_data.model[3] = {entity_.transform[3], 1.0f};
  instance_data.inv_model = glm::inverse(instance_data.model);
  instance_data.normal_matrix = glm::transpose(instance_data.inv_model);
  instance_buffer_->UploadData(&instance_data, sizeof(InstanceData));
  material_->Sync();

  scene->RegisterRenderCallback([this](graphics::CommandContext *cmd_ctx, graphics::Buffer *camera_buffer) {
    cmd_ctx->CmdBindProgram(render_program_.get());
    cmd_ctx->CmdBindResources(0, {camera_buffer}, graphics::BIND_POINT_GRAPHICS);
    cmd_ctx->CmdBindResources(1, {instance_buffer_.get()}, graphics::BIND_POINT_GRAPHICS);
    material_->BindMaterialResources(cmd_ctx);
    geometry_->DispatchDrawCalls(cmd_ctx);
  });

  if (entity_.raster_light) {
    glm::vec3 emission = material_->Emission();
    if (emission.r > 0.0f || emission.g > 0.0f || emission.b > 0.0f) {
      glm::vec4 centric_area = geometry_->CentricArea(entity_.transform);
      point_light_data.emission = emission * glm::max(0.0f, centric_area.w) * 4.0f * glm::pi<float>();
      point_light_data.position = {centric_area.x, centric_area.y, centric_area.z};
      point_light_buffer_->UploadData(&point_light_data, sizeof(PointLightData));
      scene->RegisterLightingCallback([this](graphics::CommandContext *cmd_ctx) {
        cmd_ctx->CmdBindProgram(point_light_program_.get());
        cmd_ctx->CmdBindResources(3, {point_light_buffer_.get()}, graphics::BIND_POINT_GRAPHICS);
        cmd_ctx->CmdDraw(6, 1, 0, 0);
      });
    }
  }
}

}  // namespace sparkium::raster
