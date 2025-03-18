#include "snow_mount/visualizer/visualizer_entity.h"

#include "snow_mount/visualizer/visualizer_core.h"
#include "snow_mount/visualizer/visualizer_mesh.h"
#include "visualizer_camera.h"
#include "visualizer_ownership_holder.h"
#include "visualizer_render_context.h"

namespace snow_mount::visualizer {

namespace {
#include "built_in_shaders.inl"
}

Entity::Entity(const std::shared_ptr<Core> &core) : core_(core) {
}

int Entity::ExecuteStage(RenderStage render_stage, const RenderContext &ctx) {
  return 0;
}

void Entity::PyBind(pybind11::module &m) {
  pybind11::class_<Entity, std::shared_ptr<Entity>> entity(m, "Entity");

  EntityMeshObject::PyBind(m);
}

EntityMeshObject::EntityMeshObject(const std::shared_ptr<Core> &core,
                                   const std::weak_ptr<Mesh> &mesh,
                                   const Material &material,
                                   const Matrix4<float> &transform)
    : Entity(core), mesh_(mesh) {
  program_ = core_->LoadProgram<ProgramNoNormal>(PROGRAM_ID_NO_NORMAL, [&]() {
    std::shared_ptr<ProgramNoNormal> program = std::make_shared<ProgramNoNormal>();
    core_->GraphicsCore()->CreateShader(GetShaderCode("shaders/entity.hlsl"), "VSMain", "vs_6_0",
                                        &program->vertex_shader_);
    core_->GraphicsCore()->CreateShader(GetShaderCode("shaders/entity.hlsl"), "GSMain", "gs_6_0",
                                        &program->geometry_shader_);
    core_->GraphicsCore()->CreateShader(GetShaderCode("shaders/entity.hlsl"), "PSMain", "ps_6_0",
                                        &program->fragment_shader_);
    core_->GraphicsCore()->CreateProgram(
        {FilmChannelImageFormat(FILM_CHANNEL_EXPOSURE), FilmChannelImageFormat(FILM_CHANNEL_ALBEDO),
         FilmChannelImageFormat(FILM_CHANNEL_POSITION), FilmChannelImageFormat(FILM_CHANNEL_NORMAL)},
        FilmChannelImageFormat(FILM_CHANNEL_DEPTH), &program->program_);
    program->program_->BindShader(program->vertex_shader_.get(), graphics::SHADER_TYPE_VERTEX);
    program->program_->BindShader(program->geometry_shader_.get(), graphics::SHADER_TYPE_GEOMETRY);
    program->program_->BindShader(program->fragment_shader_.get(), graphics::SHADER_TYPE_FRAGMENT);
    program->program_->AddInputBinding(sizeof(Vertex), false);
    program->program_->AddInputAttribute(0, graphics::INPUT_TYPE_FLOAT3, offsetof(Vertex, position));
    program->program_->AddInputAttribute(0, graphics::INPUT_TYPE_FLOAT3, offsetof(Vertex, normal));
    program->program_->AddInputAttribute(0, graphics::INPUT_TYPE_FLOAT2, offsetof(Vertex, tex_coord));
    program->program_->AddInputAttribute(0, graphics::INPUT_TYPE_FLOAT4, offsetof(Vertex, color));
    program->program_->AddResourceBinding(graphics::RESOURCE_TYPE_UNIFORM_BUFFER,
                                          1);  // Camera Info
    program->program_->AddResourceBinding(graphics::RESOURCE_TYPE_UNIFORM_BUFFER,
                                          1);  // Entity Info
    program->program_->SetCullMode(graphics::CULL_MODE_NONE);
    program->program_->Finalize();
    return program;
  });
  info_.material = material;
  info_.model = EigenToGLM(transform);
  core_->GraphicsCore()->CreateBuffer(sizeof(EntityInfo), graphics::BUFFER_TYPE_DYNAMIC, &info_buffer_);
}

int EntityMeshObject::ExecuteStage(RenderStage render_stage, const RenderContext &ctx) {
  std::shared_ptr<Mesh> mesh = mesh_.lock();
  if (mesh) {
    if (render_stage == RENDER_STAGE_RASTER_GEOMETRY_PASS) {
      ctx.ownership_holder->AddMesh(mesh);
      ctx.cmd_ctx->CmdBindProgram(program_->program_.get());
      ctx.cmd_ctx->CmdSetPrimitiveTopology(graphics::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
      ctx.cmd_ctx->CmdBindResources(0, {ctx.camera_buffer});
      info_buffer_->UploadData(&info_, sizeof(info_));
      ctx.cmd_ctx->CmdBindResources(1, {info_buffer_.get()});
      ctx.cmd_ctx->CmdBindVertexBuffers(0, {mesh->GetVertexBuffer()}, {0});
      ctx.cmd_ctx->CmdBindIndexBuffer(mesh->GetIndexBuffer(), 0);
      ctx.cmd_ctx->CmdDrawIndexed(mesh->IndexCount(), 1, 0, 0, 0);
    }
  } else {
    return -1;
  }
  return 0;
}

void EntityMeshObject::SetMesh(const std::shared_ptr<Mesh> &mesh) {
  mesh_ = mesh;
}

void EntityMeshObject::SetMaterial(const Material &material) {
  info_.material = material;
}

void EntityMeshObject::SetTransform(const Matrix4<float> &transform) {
  info_.model = EigenToGLM(transform);
}

void EntityMeshObject::PyBind(pybind11::module &m) {
  pybind11::class_<EntityMeshObject, Entity, std::shared_ptr<EntityMeshObject>> entity_mesh_object(m,
                                                                                                   "EntityMeshObject");
  entity_mesh_object.def("set_mesh", &EntityMeshObject::SetMesh, pybind11::arg("mesh") = nullptr);
  entity_mesh_object.def("set_material", &EntityMeshObject::SetMaterial, pybind11::arg("material") = Material{});
  entity_mesh_object.def("set_transform", &EntityMeshObject::SetTransform,
                         pybind11::arg("transform") = Matrix4<float>::Identity());
}

}  // namespace snow_mount::visualizer
