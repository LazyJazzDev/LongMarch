#include "sparkium/pipelines/raster/entity/entity_geometry_material.h"

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
  render_program_->Finalize();
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
    cmd_ctx->CmdBindResources(2, {material_->Buffer()}, graphics::BIND_POINT_GRAPHICS);
    geometry_->DispatchDrawCalls(cmd_ctx);
  });
}

}  // namespace sparkium::raster
